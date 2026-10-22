"""Table 3's intervals (#81): the delta method on the aggregates' covariance, the
finite-difference σ from posterior draws, the aggregate weights and #99's family."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "report"))
import table3_intervals as T  # noqa: E402

_C = np.array([[4.0, 1.5], [1.5, 9.0]])


def test_delta_sigma_is_exact_for_a_linear_function():
    s = T.delta_sigma(lambda x: 2.0 * x[0] - x[1], np.array([3.0, 5.0]), _C)
    g = np.array([2.0, -1.0])
    assert s == pytest.approx(np.sqrt(g @ _C @ g), rel=1e-8)


def test_delta_sigma_of_a_ratio_is_the_textbook_formula():
    x, y = 3.0, 12.0
    r = x / y
    ref = np.sqrt((_C[0, 0] - 2 * r * _C[0, 1] + r ** 2 * _C[1, 1]) / y ** 2)
    assert T.delta_sigma(lambda v: v[0] / v[1], np.array([x, y]), _C) == pytest.approx(ref, rel=1e-6)


def test_delta_sigma_of_the_hhi_uses_its_gradient():
    """∂HHI/∂x_j = (2/X)(s_j − HHI)."""
    rng = np.random.default_rng(81)
    x = np.array([40.0, 25.0, 20.0, 15.0])
    A = rng.standard_normal((4, 4))
    C = A @ A.T
    s, h = x / x.sum(), T.hhi(x)
    g = 2.0 / x.sum() * (s - h)
    assert h == pytest.approx((s ** 2).sum())
    assert T.delta_sigma(T.hhi, x, C) == pytest.approx(np.sqrt(g @ C @ g), rel=1e-6)


def test_fd_sigma_is_the_draws_second_moment_of_a_linear_function():
    rng = np.random.default_rng(81)
    D = rng.standard_normal((50, 6))
    w = np.array([1.0, 0.0, 2.0, 0.0, -1.0, 0.5])
    a = np.arange(6, dtype=float)
    s = T.fd_sigma(lambda d: w @ (a + d), D)
    assert s == pytest.approx(np.sqrt(np.mean((D @ w) ** 2)), rel=1e-6)


def test_build_spec_puts_each_mask_on_its_edge_ids():
    edges = pd.DataFrame({"edge_id": [5, 2, 7]})
    cols = [("a", np.array([True, False, True])), ("b", np.array([False, True, False]))]
    spec = T.build_spec(edges, cols, n_edges=10)
    W = spec.W.toarray()
    assert spec.names == ["a", "b"] and W.shape == (10, 2)
    np.testing.assert_array_equal(np.flatnonzero(W[:, 0]), [5, 7])
    np.testing.assert_array_equal(np.flatnonzero(W[:, 1]), [2])


def test_positions_raise_on_an_edge_outside_the_year():
    np.testing.assert_array_equal(T.positions(np.array([3, 9, 4]), np.array([4, 3])), [2, 0])
    with pytest.raises(ValueError, match="not in the year"):
        T.positions(np.array([3, 9, 4]), np.array([5]))


def test_evaluate_scales_by_birge_and_adds_quadrature_on_linear_rows_only():
    moments = SimpleNamespace(names=["x", "y"], value=np.array([3.0, 12.0]),
                              cov_post=_C, sigma_quadrature=np.array([2.5, 3.5]))
    qs = [("retention", "sum", ["x", "y"], lambda v: (v[0] + v[1]) / 10.0, True),
          ("B1", "ratio", ["x", "y"], lambda v: v[0] / v[1], False)]
    out = T.evaluate(qs, moments, scale=2.0).set_index("quantity")
    s_sum = np.sqrt(_C.sum()) / 10.0
    assert out.loc["sum", "value"] == pytest.approx(1.5)
    assert out.loc["sum", "sigma"] == pytest.approx(s_sum, rel=1e-8)
    assert out.loc["sum", "sigma_birge"] == pytest.approx(2.0 * s_sum, rel=1e-8)
    assert out.loc["sum", "hi_birge"] - out.loc["sum", "value"] == pytest.approx(
        1.959964 * 2.0 * s_sum, rel=1e-5)
    assert out.loc["sum", "sigma_quadrature"] == pytest.approx(np.hypot(2.5, 3.5) / 10.0)
    assert np.isnan(out.loc["ratio", "sigma_quadrature"])


def test_the_degenerate_family_is_cross_region_cathode_use_at_a_zero_prior_with_a_bound():
    edges = pd.DataFrame({
        "edge_id":     [0, 1, 2, 3, 4],
        "sector_from": ["Cathode", "Cathode", "Cathode", "Cathode", "CP"],
        "sector_to":   ["FU", "FU", "FU", "FU", "Cathode"],
        "region_from": ["Chile", "Chile", "Chile", "China", "Chile"],
        "region_to":   ["China", "China", "Chile", "Japan", "Chile"],
        "prior":       [0.0, 0.0, 0.0, 5.0, 0.0],
    })
    # lower-bound rows on edges 0, 2, 3, 4; edge 1 has none
    G = sp.csr_matrix((-np.ones(4), ([0, 1, 2, 3], [0, 2, 3, 4])), shape=(5, 5))
    rm = pd.DataFrame({"type": ["bound"] * 4 + ["massbal"],
                       "variant": ["lower"] * 4 + ["massbal"]})
    fam = T.degenerate_family(edges, SimpleNamespace(row_meta=rm, G=G), n_edges=5)
    np.testing.assert_array_equal(fam, [True, False, False, False, False])
