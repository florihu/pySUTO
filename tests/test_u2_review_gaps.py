"""
test_u2_review_gaps.py
======================
Tests for the unit-audit gaps of aggregation and imputation (#167, fixed in #282): the
share imputer's interpolation and boundary carries, the time imputer's non-integer
labels, id-list scope and one-sided fallbacks, the params resolver's mask path, the
ParamRow raises, the flat prior's noisy-zero mode and argument checks, and the
``AggDiagnostic`` snapshot contract.
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pysuto.relation.param_relation as pr
from pysuto.impute import (FlatPriorImputer, ImputationEngine, ShareImputer,
                           StockLeg, TimeImputer, run_imputer)
from pysuto.relation.param_relation import ParamRelationResolver
from pysuto.relation.params import SELECTOR_KEYS, ParamRow, read_params

from test_imputation import MockAcc, _agg_result, _fill
from test_param_relation import MockAcc as ResolverAcc, _row


# =============================================================================
# ShareImputer: interior interpolation and boundary carries (U2-G6-1)
# =============================================================================

class _ShareAcc:
    """One region, five years: PSC net CP→PSC (edges 0-4), base CSXEW→CP (edges 5-9)."""
    YEARS = ["1995", "1996", "1997", "1998", "1999"]

    def __init__(self):
        self.node_index = pd.DataFrame(
            {"sector": ["CP", "PSC", "CSXEW"], "entity": ["Process", "SB", "Flow"],
             "region": ["A"] * 3})
        self.edge_index = pd.DataFrame(
            {"edge_id": list(range(10)), "from_id": [0] * 5 + [2] * 5,
             "to_id": [1] * 5 + [0] * 5, "time": self.YEARS * 2, "valuation": ["Mass"] * 10})
        self.n_edges = 10


_PSC = StockLeg("transfer", ("CP", "Process", "PSC", "SB"),
                (("CSXEW", "Flow", "CP", "Process"),))


def _share(base, net):
    acc = _ShareAcc()
    f, sigma = np.full(10, np.nan), np.full(10, np.nan)
    f[5:] = base
    for i, v in net.items():
        f[i] = v
    imp = ShareImputer(p95={("PSC", "A"): 0.05}, legs=[_PSC], scale_abs_floor=1.0)
    run_imputer(imp, f, sigma, 0, acc)
    return f


def test_the_observed_share_is_interpolated_inside_and_carried_beyond_the_observed_years():
    # shares: 1996 = 2/100 = 0.02, 1998 = 8/200 = 0.04
    f = _share([100.0, 100.0, 150.0, 200.0, 300.0], {1: 2.0, 3: 8.0})
    assert f[0] == pytest.approx(0.02 * 100.0, rel=1e-12)          # NOCB before the first
    assert f[2] == pytest.approx(0.03 * 150.0, rel=1e-12)          # linear between the two
    assert f[4] == pytest.approx(0.04 * 300.0, rel=1e-12)          # LOCF after the last
    assert f[1] == 2.0 and f[3] == 8.0                              # observed stay


def test_a_year_without_a_base_is_left_missing():
    f = _share([100.0, 100.0, np.nan, 200.0, 300.0], {1: 2.0, 3: 8.0})
    assert np.isnan(f[2])


def test_a_zero_base_gives_no_share_and_the_leg_is_left_to_the_flat_prior():
    f = _share([0.0, 0.0, 0.0, 0.0, 0.0], {1: 2.0, 3: 8.0})
    assert np.isnan(f[0]) and np.isnan(f[2]) and np.isnan(f[4])


# =============================================================================
# TimeImputer: labels, id-list scope, one-sided fallbacks (U2-G6-2)
# =============================================================================

class _TimeAcc:
    """Series 0→1 and 2→3, one edge per label in ``times``; edge ids run series by series."""

    def __init__(self, times, series=((0, 1), (2, 3))):
        rows = []
        for fi, ti in series:
            for t in times:
                rows.append((len(rows), fi, ti, t, "Mass"))
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"])
        self.n_edges = len(rows)


def _time(acc, f, **kw):
    f = np.array(f, dtype=float)
    sigma = np.where(np.isnan(f), np.nan, 1.0)
    entries = run_imputer(TimeImputer(**kw), f, sigma, 0, acc)
    return f, sigma, entries


def test_non_integer_time_labels_sort_by_text_and_interpolate():
    f, _, entries = _time(_TimeAcc(["a", "b", "c"], series=((0, 1),)), [10.0, np.nan, 20.0])
    assert f[1] == pytest.approx(15.0, rel=1e-12)
    assert entries[0]["method"] == "time_linear"


def test_an_id_list_scope_edges_restricts_fills_to_those_series():
    acc = _TimeAcc(["2020", "2021", "2022"])
    f, _, _ = _time(acc, [10.0, np.nan, 20.0, 30.0, np.nan, 40.0], scope_edges=[0, 1, 2])
    assert f[1] == pytest.approx(15.0, rel=1e-12)
    assert np.isnan(f[4])


def test_linear_with_one_observation_before_carries_it_and_inflates_sigma_by_the_gap():
    acc = _TimeAcc(["2020", "2021", "2022"], series=((0, 1),))
    f, sigma, entries = _time(acc, [10.0, np.nan, np.nan], gap_inflation=1.5)
    assert f[1] == 10.0 and f[2] == 10.0
    assert sigma[1] == pytest.approx(1.5, rel=1e-12)
    assert sigma[2] == pytest.approx(1.5 ** 2, rel=1e-12)
    assert {e["method"] for e in entries} == {"time_linear"}


def test_linear_with_one_observation_after_carries_it_backward():
    acc = _TimeAcc(["2020", "2021", "2022"], series=((0, 1),))
    f, sigma, _ = _time(acc, [np.nan, np.nan, 10.0], gap_inflation=1.5)
    assert f[0] == 10.0 and f[1] == 10.0
    assert sigma[0] == pytest.approx(1.5 ** 2, rel=1e-12)      # the 2-year carry


def test_interpolate_only_does_not_carry():
    acc = _TimeAcc(["2020", "2021", "2022"], series=((0, 1),))
    f, _, _ = _time(acc, [10.0, np.nan, np.nan], interpolate_only=True)
    assert np.isnan(f[1]) and np.isnan(f[2])


# =============================================================================
# Params resolver: the mask path equals the enumeration path (U2-G6-4)
# =============================================================================

@pytest.mark.parametrize("sel", [
    {},
    {"a_sector_from": "Mining"},
    {"a_valuation": "Mass", "a_region_from": "A"},
    {"a_sector_from": "Mining", "a_region_to": "same"},
    {"a_sector_from": "Nowhere"},
])
def test_the_vectorised_mask_path_matches_enumeration(sel, monkeypatch):
    acc = ResolverAcc()
    row = _row(**sel)
    by_enum = sorted(ParamRelationResolver(acc).match_a_edges(row))
    monkeypatch.setattr(pr, "A_PRODUCT_THRESHOLD", 0)
    by_mask = sorted(ParamRelationResolver(acc).match_a_edges(row))
    assert by_mask == by_enum


# =============================================================================
# ParamRow raises (U2-G6-4)
# =============================================================================

def _params_csv(tmp_path, **kw):
    rec = dict.fromkeys(SELECTOR_KEYS, "*")
    rec.update(param_id="p", param_type="concentration", value_nominal="0.5",
               unc_type="sd", value_sd="0.05", value_min="", value_max="", epsilon="0")
    for c in ("pedigree_reliability", "pedigree_completeness",
              "pedigree_temporal_correlation", "pedigree_geographical_correlation",
              "pedigree_other_correlation"):
        rec[c] = ""
    rec.update(kw)
    path = tmp_path / "p.csv"
    pd.DataFrame([rec]).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("kw, match", [
    ({"value_sd": ""}, "needs value_sd"),
    ({"unc_type": "min_max", "value_sd": "", "value_min": "0.1"}, "needs value_min"),
    ({"unc_type": "pedigree", "value_nominal": ""}, "needs value_nominal"),
    ({"unc_type": "pedigree", "pedigree_reliability": "x"}, "not five integers"),
    ({"unc_type": "bogus"}, "is not one of"),
])
def test_an_unusable_uncertainty_cell_raises_at_read(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        read_params(_params_csv(tmp_path, **kw), {"concentration"})


# =============================================================================
# FlatPriorImputer: noisy-zero mode and argument raises (U2-G6-4)
# =============================================================================

def test_noisy_zero_fills_every_missing_edge_with_zero_at_sigma_high():
    acc = MockAcc()
    f, sigma = np.full(10, np.nan), np.full(10, np.nan)
    f[0], sigma[0] = 5.0, 0.5
    entries = run_imputer(FlatPriorImputer(acc, mode="noisy_zero", sigma_high=7.0),
                          f, sigma, 0, acc)
    assert f[0] == 5.0 and np.all(f[1:] == 0.0) and np.all(sigma[1:] == 7.0)
    assert {e["method"] for e in entries} == {"flat_prior_noisy_zero"}


@pytest.mark.parametrize("kw, match", [
    ({"mode": "bogus"}, "unknown mode"),
    ({"mode": "noisy_zero"}, "sigma_high > 0"),
    ({"mode": "noisy_zero", "sigma_high": 0.0}, "sigma_high > 0"),
    ({}, "requires both"),
])
def test_flat_prior_argument_errors(kw, match):
    with pytest.raises(ValueError, match=match):
        FlatPriorImputer(MockAcc(), **kw)


# =============================================================================
# AggDiagnostic snapshots the aggregation at construction (U2-S1-2)
# =============================================================================

def test_agg_diagnostic_keeps_reporting_the_pre_imputation_coverage(capsys):
    from pysuto.diag import AggDiagnostic
    acc = MockAcc()
    ar = _agg_result([10.0] + [np.nan] * 9, [1.0] + [np.nan] * 9)
    diag = AggDiagnostic(ar, acc_struct=acc)
    before = diag.summary()
    engine = ImputationEngine(acc_struct=acc)
    engine.register(_fill([(0, 2, 0.5, 0.0, 0.0, "p", "d")]))
    result = engine.run(ar)
    assert int(np.isfinite(result.f_agg).sum()) == 2          # imputation did fill edge 2
    assert "covered edges: 1 / 10" in before
    assert diag.summary() == before
    capsys.readouterr()
