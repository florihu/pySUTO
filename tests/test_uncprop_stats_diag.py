"""
test_uncprop_stats_diag.py
==========================
Tests for :class:`pysuto.diag.stats.UncPropStatsDiagnostic` — the Stage-5 edge
frame (posterior contraction κ, defect z, weight, prior class, κ–z quadrant,
bound-active flag) and its aggregates.

Synthetic stubs only: 6 edges over two years (1995: e0–e2, 1996: e3–e5) on the
block path (year-local arrays over ``cols_t``). Every number is hand-checkable:

    σ_a  = [2, 1, 1 | 2, 1, 1]          σ_a* = [1, 1, 0.5 | 2, 0, 1]
    κ    = [0.75, 0, 0.75 | 0, 1, 0]
    a_prior = [10, 4, 0 | 6, 5, 1]      a*   = [2, 4.5, 3 | 6, 0, 9]
    z    = [−4, 0.5, 3 | 0, −5, 8]
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pysuto.diag.stats import UncPropStatsDiagnostic
from pysuto.diag.stats.uncprop import classify_priors, quadrant


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _cr(sigma_a=(2, 1, 1, 2, 1, 1)):
    """ConstraintRun stub: lower-bound rows on e1 and e4, one mass balance."""
    G = sp.csr_matrix(np.array([
        [0, -1, 0, 0,  0, 0],      # bound/lower on e1 (1995)
        [0,  0, 0, 0, -1, 0],      # bound/lower on e4 (1996)
        [1, -1, 0, 0,  0, 0],      # mass balance (1995)
    ], dtype=float))
    return SimpleNamespace(
        G        = G,
        row_meta = pd.DataFrame({
            "type":    ["bound", "bound", "massbal"],
            "variant": ["lower", "lower", "massbal"],
            "time":    ["1995", "1996", "1995"],
        }),
        a0      = np.array([10, 4, 0, 6, 5, 1], dtype=float),
        sigma_a = np.asarray(sigma_a, dtype=float),
        l       = np.zeros(6),
        u       = np.array([100, 100, 100, 0, 100, 100], dtype=float),   # e3 zero-pinned
    )


def _unc():
    ts = {
        "1995": SimpleNamespace(sigma_a_star=np.array([1.0, 1.0, 0.5]),
                                lower=np.array([0.0, 2.5, 2.0]),
                                upper=np.array([4.0, 6.5, 4.0]),
                                cols_t=np.array([0, 1, 2])),
        "1996": SimpleNamespace(sigma_a_star=np.array([2.0, 0.0, 1.0]),
                                lower=np.array([2.0, 0.0, 7.0]),
                                upper=np.array([10.0, 0.0, 11.0]),
                                cols_t=np.array([3, 4, 5])),
    }
    return SimpleNamespace(method="kalman", alpha=0.05, per_timestep=ts)


def _opt():
    return {
        "1995": SimpleNamespace(F_opt=np.array([2.0, 4.5, 3.0]),
                                a_prior=np.array([10.0, 4.0, 0.0]),
                                cols_t=np.array([0, 1, 2])),
        "1996": SimpleNamespace(F_opt=np.array([6.0, 0.0, 9.0]),
                                a_prior=np.array([6.0, 5.0, 1.0]),
                                cols_t=np.array([3, 4, 5])),
    }


class _Log:
    def __init__(self, df):
        self._df = df

    def as_dataframe(self):
        return self._df


def _imp():
    log = pd.DataFrame({
        "edge_id":    [1, 2, 3, 4, 5],
        "imputer_id": ["network", "flat_prior", "flat_prior", "network", "trade"],
        "method":     ["network_forward", "flat_prior_noisy_zero", "flat_prior_noisy_zero",
                       "network_stripping", "trade_seed"],
    })
    return SimpleNamespace(coverage=np.array([1, 0, 0, 0, 0, 0], dtype=bool),
                           imputation_log=_Log(log))


class _Acc:
    edge_template_code = np.array([7, 8, 9, 7, 8, 9])

    @property
    def edge_meta(self):
        return pd.DataFrame({
            "edge_id":     list(range(6)),
            "time":        ["1995"] * 3 + ["1996"] * 3,
            "valuation":   ["Mass", "Mass", "CC"] * 2,
            "from_sector": ["Mining", "Conc", "Conc"] * 2,
            "from_region": ["A", "A", "B"] * 2,
            "to_sector":   ["Conc", "Smelt", "Smelt"] * 2,
            "to_region":   ["A", "B", "B"] * 2,
        })


@pytest.fixture
def frame():
    d = UncPropStatsDiagnostic(_unc(), _opt(), _cr(), _Acc(), imp_result=_imp())
    return d.edge_frame().set_index(["year", "edge_id"])


# ---------------------------------------------------------------------------
# Edge frame values
# ---------------------------------------------------------------------------

def test_kappa_is_one_minus_variance_ratio(frame):
    np.testing.assert_allclose(frame["kappa"].to_numpy(),
                               [0.75, 0.0, 0.75, 0.0, 1.0, 0.0], rtol=1e-12)


def test_z_is_standardised_defect(frame):
    np.testing.assert_allclose(frame["z"].to_numpy(),
                               [-4.0, 0.5, 3.0, 0.0, -5.0, 8.0], rtol=1e-12)


def test_weight_keeps_deleted_and_inflated_edges(frame):
    # e0 deleted 10 → 2 keeps 10; e2 inflated 0 → 3 keeps 3; e4 deleted to 0 keeps 5
    np.testing.assert_allclose(frame["weight"].to_numpy(),
                               [10.0, 4.5, 3.0, 6.0, 5.0, 9.0], rtol=1e-12)


def test_quadrants_with_inclusive_cuts(frame):
    assert list(frame["quadrant"]) == [
        "overruled", "prior-carried", "confirmed",          # e2: |z| = 3 is not > 3
        "prior-carried", "overruled", "unexplained move",
    ]


def test_quadrant_kappa_cut_is_inclusive():
    assert list(quadrant([0.5, 0.4999], [0.0, 0.0])) == ["confirmed", "prior-carried"]


def test_bound_active_needs_a_lower_bound_row_and_zero_posterior(frame):
    # e4 has a bound row and a* = 0; e1 has a bound row but a* = 4.5;
    # e3 has a* = 6 and no row.
    np.testing.assert_array_equal(frame["bound_active"].to_numpy(),
                                  [False, False, False, False, True, False])


def test_bound_active_uses_the_runs_bind_tol():
    d = UncPropStatsDiagnostic(_unc(), _opt(), _cr(), conditioning={"bind_tol": 5.0})
    f = d.edge_frame().set_index("edge_id")
    assert f.loc[1, "bound_active"] and f.loc[4, "bound_active"]     # a* = 4.5 and 0


def test_prior_classes(frame):
    assert list(frame["prior_class"]) == [
        "from a feed", "imputed", "flat", "zero-pinned", "flat", "imputed",
    ]


def test_persisted_bands_and_edge_meta_carried(frame):
    np.testing.assert_allclose(frame["lower"].to_numpy(), [0, 2.5, 2, 2, 0, 7], rtol=1e-12)
    np.testing.assert_array_equal(frame["identity"].to_numpy(), [7, 8, 9, 7, 8, 9])
    assert frame.loc[("1995", 2), "label"] == "Conc|B→Smelt|B|CC"


# ---------------------------------------------------------------------------
# Fallbacks and conditioning
# ---------------------------------------------------------------------------

def test_without_imp_and_acc_every_edge_is_one_class_and_valuation():
    f = UncPropStatsDiagnostic(_unc(), _opt(), _cr()).edge_frame()
    assert set(f["prior_class"]) == {"all"}
    assert set(f["valuation"]) == {"all"}
    np.testing.assert_array_equal(f["identity"].to_numpy(), [0, 1, 2, 3, 4, 5])


def test_sigma_a_is_conditioned_with_the_given_floor():
    cr = _cr(sigma_a=(0.1, 1, 1, 2, 1, 1))
    f = UncPropStatsDiagnostic(_unc(), _opt(), cr,
                               conditioning={"sigma_a_floor_abs": 0.5}).edge_frame()
    assert f.loc[f["edge_id"] == 0, "sigma_a"].item() == pytest.approx(0.5)


def test_legacy_full_length_arrays_cover_every_edge():
    unc = SimpleNamespace(alpha=0.05, per_timestep={"1995": SimpleNamespace(
        sigma_a_star=np.ones(6), lower=np.zeros(6), upper=np.ones(6), cols_t=None)})
    opt = {"1995": SimpleNamespace(F_opt=np.arange(6, dtype=float), a_prior=None,
                                   cols_t=None)}
    f = UncPropStatsDiagnostic(unc, opt, _cr()).edge_frame()
    np.testing.assert_array_equal(f["edge_id"].to_numpy(), np.arange(6))
    np.testing.assert_allclose(f["a_prior"].to_numpy(), _cr().a0, rtol=1e-12)


def test_missing_opt_year_raises():
    with pytest.raises(KeyError, match="1996"):
        UncPropStatsDiagnostic(_unc(), {"1995": _opt()["1995"]}, _cr())


def test_a_feed_covered_edge_is_never_zero_pinned():
    log = pd.DataFrame({"edge_id": [1], "imputer_id": ["trade"], "method": ["trade_seed"]})
    out = classify_priors(np.array([True, False]), log, np.zeros(2), np.zeros(2))
    assert list(out) == ["from a feed", "zero-pinned"]


def test_misaligned_partitions_raise():
    opt = _opt()
    opt["1996"].cols_t = np.array([3, 5, 4])
    with pytest.raises(ValueError, match="1996"):
        UncPropStatsDiagnostic(_unc(), opt, _cr()).edge_frame()


def test_classify_priors_raises_on_an_unclassified_edge():
    log = pd.DataFrame({"edge_id": [1], "imputer_id": ["trade"], "method": ["trade_seed"]})
    with pytest.raises(ValueError, match="1 edge"):
        classify_priors(np.array([True, False, False]), log,
                        np.zeros(3), np.full(3, 10.0))


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def test_trace_contraction_per_year():
    d = UncPropStatsDiagnostic(_unc(), _opt(), _cr())
    tc = d.trace_contraction(d.edge_frame())
    assert tc["1995"] == pytest.approx(1 - 2.25 / 6)
    assert tc["1996"] == pytest.approx(1 - 5 / 6)


def test_contraction_summary_sums_weight_per_cell():
    d = UncPropStatsDiagnostic(_unc(), _opt(), _cr(), _Acc(), imp_result=_imp())
    s = d.contraction_summary(d.edge_frame()).set_index(
        ["year", "valuation", "prior_class", "quadrant"])
    assert s.loc[("1996", "Mass", "flat", "overruled"), "weight_kt"] == pytest.approx(5.0)
    assert s.loc[("1996", "Mass", "flat", "overruled"), "n_edges"] == 1
    assert s["n_edges"].sum() == 6
    assert s["weight_kt"].sum() == pytest.approx(37.5)
