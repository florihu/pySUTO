"""
test_icsg_feed.py
=================
The ICSG feed's observed zeros (#76, narrowed by #78): a weighted fan's domestic leg at
trade-share weight 0, in a fan the trade shares bound by the region's own production, in
a region that produces none of that product, leaves Stage 1 covered at 0, priced on its
fans' usage, and reaches Stage 2 as covered.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from icsg_feed import (non_producer_mask, observed_zero_sigma,    # noqa: E402
                       stamp_observed_zeros)
from pysuto.collector import FeedCollector                        # noqa: E402
from pysuto.conc import ConcordanceTable, WeightMatrix            # noqa: E402
from pysuto.unc import PropagationResult                          # noqa: E402
from test_collector import _MockAccStruct, _MockRegistry          # noqa: E402
from test_conc import maps, mock_acc_struct, raw_use_only         # noqa: E402,F401

DOMESTIC, IMPORT = 6, 7          # Concentrate(A) → Refining(A), Concentrate(B) → Refining(A)


def _legs(*pairs):
    return pd.DataFrame(pairs, columns=["src_id", "edge_id"])


# ---------------------------------------------------------------------------
# observed_zero_sigma
# ---------------------------------------------------------------------------

def test_a_small_fan_prices_its_zero_at_the_floor():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([10.0]), np.zeros(3))
    assert sigma[1] == pytest.approx(0.3)
    assert np.isnan(sigma[[0, 2]]).all()


def test_a_large_fan_prices_its_zero_at_one_percent_of_its_usage():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([100.0]), np.zeros(3))
    assert sigma[1] == pytest.approx(1.0)


def test_a_leg_another_fan_feeds_is_not_an_observed_zero():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([100.0]), np.array([0.0, 2.5, 0.0]))
    assert np.isnan(sigma).all()


def test_a_leg_several_fans_weigh_at_zero_is_priced_on_their_summed_usage():
    # an aggregate region (ROW) fans each member country's usage onto the same legs
    sigma = observed_zero_sigma(_legs((0, 1), (1, 1)), np.array([50.0, 400.0]), np.zeros(2))
    assert sigma[1] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# The brief's fan, from the trade-share weights to the collector
# ---------------------------------------------------------------------------

def _builder_weights(usage, imports):
    """The trade-share weights for a region using ``usage`` and importing ``imports``
    from B: the import leg carries the BACI mass and the domestic leg the residual
    max(0, usage − imports) (ADR 0002 option D)."""
    return max(0.0, usage - imports), imports


def _collected(raw_use_only, mock_acc_struct, maps, usage, w_domestic, w_import,
               producer=False):
    """Fan ``usage`` over the toy legs with these weights, stamp the observed zeros
    (A produces none of the product unless ``producer``), and load the result through
    the collector at Stage 2's min_flow_kt."""
    uw = pd.DataFrame([
        {"root_sector_from": "Concentrate", "root_sector_to": "Refining",
         "root_region_from": r, "root_region_to": "A", "root_time": 0, "weight": w}
        for r, w in (("A", w_domestic), ("B", w_import))])
    ct = ConcordanceTable.from_raw(raw_df=raw_use_only, acc_struct=mock_acc_struct,
                                   aux_col="Supply_Use", id="t", user_weights_df=uw,
                                   drop_unweighted_fans=True, **maps)
    F_src = np.full(ct.n_src_edges, usage)
    F_root = WeightMatrix(ct).apply(F_src)
    result = PropagationResult(F_root=F_root, sigma_root=0.05 * np.abs(F_root),
                               cv_root=np.full_like(F_root, 0.05), cv_src=0.05)
    n = stamp_observed_zeros(ct, F_src, result, np.full(len(F_root), not producer))
    registry = _MockRegistry({"icsg": {"F_root": result.F_root, "sigma_root": result.sigma_root,
                                       "observed_zero": result.observed_zero}})
    coll = FeedCollector(registry, _MockAccStruct(len(F_root)), min_flow_kt=0.01)
    F, Sigma = coll.select(["icsg"])._load_arrays()
    return n, F[0], Sigma[0]


def test_a_fan_whose_imports_exceed_its_usage_feeds_its_domestic_leg_as_a_zero(
        raw_use_only, mock_acc_struct, maps):
    n, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 10.0,
                             *_builder_weights(usage=10.0, imports=12.0))
    assert n == 1
    assert F[DOMESTIC] == 0.0 and Sigma[DOMESTIC] == pytest.approx(0.3)
    assert F[IMPORT] == pytest.approx(10.0)


def test_a_larger_fan_prices_its_domestic_zero_at_one_percent(raw_use_only, mock_acc_struct, maps):
    _, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 100.0,
                             *_builder_weights(usage=100.0, imports=120.0))
    assert F[DOMESTIC] == 0.0 and Sigma[DOMESTIC] == pytest.approx(1.0)


def test_a_fan_with_no_positive_weight_stays_undistributed(raw_use_only, mock_acc_struct, maps):
    # weights present but all zero: ADR 0008 drops the fan, so not even a zero is fed
    uw = pd.DataFrame([
        {"root_sector_from": "Concentrate", "root_sector_to": "Refining",
         "root_region_from": r, "root_region_to": "A", "root_time": 0, "weight": 0.0}
        for r in ("A", "B")])
    ct = ConcordanceTable.from_raw(raw_df=raw_use_only, acc_struct=mock_acc_struct,
                                   aux_col="Supply_Use", id="t", user_weights_df=uw,
                                   drop_unweighted_fans=True, **maps)
    zeros = np.zeros(ct.n_root_edges)
    result = PropagationResult(F_root=zeros, sigma_root=zeros.copy(), cv_root=zeros, cv_src=0.05)
    assert ct.tab.empty and stamp_observed_zeros(
        ct, np.zeros(0), result, np.ones(ct.n_root_edges, dtype=bool)) == 0


def test_a_producers_zero_weight_leg_stays_open_to_the_imputer(
        raw_use_only, mock_acc_struct, maps):
    """Regression (#78): Italy's, the UK's and South Africa's domestic semis legs were
    zero only because imports exceed usage; pinning them left the regions' own refined
    output nowhere to go, and common-set sum-row z² rose 29 %."""
    n, F, Sigma = _collected(raw_use_only, mock_acc_struct, maps, 10.0,
                             *_builder_weights(usage=10.0, imports=12.0), producer=True)
    assert n == 0
    assert np.isnan(F[DOMESTIC]) and np.isnan(Sigma[DOMESTIC])   # not covered: imputable


# ---------------------------------------------------------------------------
# non_producer_mask
# ---------------------------------------------------------------------------

def _edges(*rows):
    return pd.DataFrame(rows, columns=["edge_id", "root_sector_from", "root_region_to",
                                       "root_time"])


def _production(*rows):
    return pd.DataFrame(rows, columns=["root_sector_from", "root_region_to", "root_time",
                                       "production_cc"])


def test_only_a_non_producers_production_bounded_legs_may_carry_an_observed_zero():
    """Germany mines no concentrate, so its domestic concentrate leg at weight 0 is an
    observation (#32); Spain mines some, and a cathode fan has no production bound."""
    edges = _edges((0, "Concentrate", "Germany", "2018"), (1, "Concentrate", "Spain", "2018"),
                   (2, "Cathode", "Italy", "2018"), (3, "Concentrate", "Chile", "2018"))
    prod = _production(("Concentrate", "Spain", "2018", 40.0),
                       ("Concentrate", "Germany", "2018", 0.0))
    assert non_producer_mask(edges, prod, 5).tolist() == [True, False, False, True, False]
