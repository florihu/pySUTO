"""
test_param_relation.py
======================
Unit tests for pysuto.param_relation — CSV-driven A→B parametric edge
relation resolver, shared by Stage-2 ParametricImputer and Stage-3
RatioConstraintBuilder / RatioBoundBuilder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.param_relation import (
    A_FIELDS,
    A_PRODUCT_THRESHOLD,
    B_FIELDS,
    WILDCARD,
    ParamRelationResolver,
    ResolvedRelation,
    expand_edge_index,
    float_or,
    get_edge_cache,
    sigma_from_row,
    specificity,
)


# =============================================================================
# Minimal mock accounting structure (mirrors tests/test_imputation.py::MockAcc)
# =============================================================================

class MockAcc:
    """
    Small accounting structure for resolver tests.

    Nodes:
        0  Mining / Process / A
        1  Ore    / Flow    / A
        2  Mining / Process / B
        3  Ore    / Flow    / B

    Edges:
        0  0→1  t=2020  Mass    Mining/A → Ore/A
        1  2→3  t=2020  Mass    Mining/B → Ore/B
        2  0→1  t=2020  CC      same pair, CC valuation
        3  2→3  t=2020  CC      same pair, CC valuation
        4  1→0  t=2020  Mass    Ore/A → Mining/A
        5  3→2  t=2020  Mass    Ore/B → Mining/B
        6  0→1  t=2021  Mass    Mining/A → Ore/A, next year
        7  2→3  t=2021  Mass    Mining/B → Ore/B, next year
    """

    def __init__(self) -> None:
        self.node_index = pd.DataFrame(
            {
                "sector": ["Mining", "Ore", "Mining", "Ore"],
                "entity": ["Process", "Flow", "Process", "Flow"],
                "region": ["A", "A", "B", "B"],
            },
            index=[0, 1, 2, 3],
        )
        self.edge_index = pd.DataFrame(
            {
                "edge_id":   [0, 1, 2, 3, 4, 5, 6, 7],
                "from_id":   [0, 2, 0, 2, 1, 3, 0, 2],
                "to_id":     [1, 3, 1, 3, 0, 2, 1, 3],
                "time":      ["2020", "2020", "2020", "2020", "2020", "2020", "2021", "2021"],
                "valuation": ["Mass", "Mass", "CC",   "CC",   "Mass", "Mass", "Mass", "Mass"],
            }
        )
        self.n_edges = 8


@pytest.fixture
def acc():
    return MockAcc()


def _row(**kwargs) -> pd.Series:
    """Build a CSV-shaped row with wildcards on every unspecified selector."""
    base: dict = {}
    for col_key, _, _ in A_FIELDS:
        base[col_key] = "*"
    for col_key, _, _ in B_FIELDS:
        base[col_key] = "*"
    base.update(kwargs)
    return pd.Series(base)


# =============================================================================
# sigma_from_row — moved from tests/test_imputation.py::TestSigmaFromRow
# =============================================================================

class TestSigmaFromRow:

    def test_sd(self):
        row = {"unc_type": "sd", "value_sd": "0.5"}
        assert sigma_from_row(pd.Series(row), 1.0) == pytest.approx(0.5)

    def test_min_max(self):
        row = {"unc_type": "min_max", "value_min": "0.2", "value_max": "0.6"}
        # (0.6 − 0.2) / 4 = 0.1
        assert sigma_from_row(pd.Series(row), 0.4) == pytest.approx(0.1)

    def test_pedigree_positive(self):
        row = {
            "unc_type": "pedigree",
            "pedigree_reliability": "2",
            "pedigree_completeness": "2",
            "pedigree_temporal_correlation": "1",
            "pedigree_geographical_correlation": "1",
            "pedigree_other_correlation": "2",
        }
        sigma = sigma_from_row(pd.Series(row), 1.0)
        assert sigma > 0

    def test_unknown_unc_type_raises(self):
        with pytest.raises(ValueError):
            sigma_from_row(pd.Series({"unc_type": "unknown"}), 1.0)


# =============================================================================
# float_or — lenient float parse
# =============================================================================

class TestFloatOr:

    def test_returns_float_for_numeric(self):
        assert float_or("1.5", default=0.0) == 1.5

    def test_returns_default_for_blank(self):
        assert float_or("", default=0.0) == 0.0

    def test_returns_default_for_nan_string(self):
        assert np.isnan(float_or("nan", default=np.nan))

    def test_returns_default_for_none_string(self):
        assert float_or("None", default=42.0) == 42.0


# =============================================================================
# specificity — count non-wildcard A-fields
# =============================================================================

class TestSpecificity:

    def test_all_wildcards_is_zero(self):
        assert specificity(_row()) == 0

    def test_one_concrete_field_is_one(self):
        assert specificity(_row(a_sector_from="Mining")) == 1

    def test_all_concrete_is_eight(self):
        row = _row(
            a_sector_from="Mining", a_entity_from="Process", a_region_from="A",
            a_sector_to="Ore",     a_entity_to="Flow",      a_region_to="A",
            a_time="2020",         a_valuation="Mass",
        )
        assert specificity(row) == 8


# =============================================================================
# expand_edge_index — decode sector/entity/region
# =============================================================================

class TestExpandEdgeIndex:

    def test_adds_decoded_columns(self, acc):
        ei = expand_edge_index(acc.edge_index, acc.node_index)
        for col in ("sector_from", "entity_from", "region_from",
                    "sector_to",   "entity_to",   "region_to"):
            assert col in ei.columns
        # Edge 0: 0→1 = Mining/Process/A → Ore/Flow/A
        row0 = ei[ei["edge_id"] == 0].iloc[0]
        assert (row0["sector_from"], row0["entity_from"], row0["region_from"]) == ("Mining", "Process", "A")
        assert (row0["sector_to"],   row0["entity_to"],   row0["region_to"])   == ("Ore",    "Flow",    "A")


# =============================================================================
# get_edge_cache — per-acc_struct cache
# =============================================================================

class TestGetEdgeCache:

    def test_caches_per_acc_struct(self, acc):
        c1 = get_edge_cache(acc)
        c2 = get_edge_cache(acc)
        assert c1 is c2  # identity preserved across calls

    def test_returns_six_components(self, acc):
        ei_exp, edge_lookup, id_to_tuple, field_vals, mask_cache, edge_ids_arr = get_edge_cache(acc)
        assert len(ei_exp) == acc.n_edges
        assert len(edge_lookup) == acc.n_edges
        assert len(id_to_tuple) == acc.n_edges
        assert "sector_from" in field_vals
        assert edge_ids_arr.shape == (acc.n_edges,)


# =============================================================================
# ParamRelationResolver — the resolver itself
# =============================================================================

class TestResolverExplicitA:

    def test_single_explicit_a_match(self, acc):
        """Fully-pinned A-selector resolves to exactly one A edge."""
        resolver = ParamRelationResolver(acc)
        # A: Mining/Process/A → Ore/Flow/A at t=2020, Mass → edge 0 only
        row = _row(
            a_sector_from="Mining", a_entity_from="Process", a_region_from="A",
            a_sector_to="Ore",      a_entity_to="Flow",      a_region_to="A",
            a_time="2020",          a_valuation="Mass",
            # B fully wildcard ⇒ every edge in the index qualifies
        )
        results = list(resolver.resolve_row(row))
        a_ids = {r.a_edge_id for r in results}
        assert a_ids == {0}
        # Specificity = 8 non-wildcard A-fields
        assert all(r.specificity == 8 for r in results)


class TestResolverWildcards:

    def test_wildcard_a_region_matches_both_regions(self, acc):
        """Wildcard on a_region_from expands across both regions."""
        resolver = ParamRelationResolver(acc)
        row = _row(
            a_sector_from="Mining", a_entity_from="Process",
            a_sector_to="Ore",      a_entity_to="Flow",
            a_time="2020",          a_valuation="Mass",
            # a_region_from / a_region_to wildcard
            b_sector_from="Mining", b_entity_from="Process",  # restrict B too
            b_sector_to="Ore",      b_entity_to="Flow",
            b_time="2020",          b_valuation="Mass",
            b_region_from="same",   b_region_to="same",
        )
        results = list(resolver.resolve_row(row))
        a_ids = {r.a_edge_id for r in results}
        # Should hit both region-A and region-B at t=2020, Mass: edges 0 and 1
        assert a_ids == {0, 1}


class TestResolverSameDirective:

    def test_a_region_to_same_couples_regions(self, acc):
        """'same' on a_region_to constrains region_to=region_from."""
        resolver = ParamRelationResolver(acc)
        row = _row(
            a_sector_from="Mining", a_entity_from="Process",
            a_sector_to="Ore",      a_entity_to="Flow",
            a_region_to="same",   # tie region_to to region_from
            a_time="2020",          a_valuation="Mass",
        )
        results = list(resolver.resolve_row(row))
        # Both A-region values get matched but only if region_to == region_from.
        # Edges 0 (A→A) and 1 (B→B) qualify; cross-region edges do not exist
        # in the fixture so this just confirms both intra-region edges resolve.
        a_ids = {r.a_edge_id for r in results}
        assert a_ids == {0, 1}


class TestResolverBSideSame:

    def test_b_sector_from_same_constrains_b_to_a(self, acc):
        """B with all 'same' should resolve B to the same edge as A."""
        resolver = ParamRelationResolver(acc)
        row = _row(
            a_sector_from="Mining", a_entity_from="Process", a_region_from="A",
            a_sector_to="Ore",      a_entity_to="Flow",      a_region_to="A",
            a_time="2020",          a_valuation="Mass",
            b_sector_from="same",   b_entity_from="same",    b_region_from="same",
            b_sector_to="same",     b_entity_to="same",      b_region_to="same",
            b_time="same",          b_valuation="same",
        )
        results = list(resolver.resolve_row(row))
        assert len(results) == 1
        assert results[0].a_edge_id == results[0].b_edge_id == 0


class TestResolverNoMatch:

    def test_selector_with_no_matching_edge_yields_nothing(self, acc):
        resolver = ParamRelationResolver(acc)
        row = _row(
            a_sector_from="Nonexistent",  # no such sector in the fixture
            a_entity_from="Process",      a_region_from="A",
            a_sector_to="Ore",            a_entity_to="Flow",      a_region_to="A",
            a_time="2020",                a_valuation="Mass",
        )
        results = list(resolver.resolve_row(row))
        assert results == []


class TestResolverRowPassthrough:

    def test_resolved_relation_carries_originating_row(self, acc):
        resolver = ParamRelationResolver(acc)
        row = _row(
            a_sector_from="Mining", a_entity_from="Process", a_region_from="A",
            a_sector_to="Ore",      a_entity_to="Flow",      a_region_to="A",
            a_time="2020",          a_valuation="Mass",
            value_nominal="2.5", unc_type="sd", value_sd="0.1",
        )
        results = list(resolver.resolve_row(row))
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, ResolvedRelation)
            # Resolver yields the same pd.Series object → adapters can read extra fields
            assert r.row["value_nominal"] == "2.5"
            assert r.row["unc_type"] == "sd"


class TestResolverBValuation:

    def test_b_valuation_picks_specific_edge(self, acc):
        """B-side concrete valuation routes A=Mass edge to B=CC counterpart."""
        resolver = ParamRelationResolver(acc)
        row = _row(
            a_sector_from="Mining", a_entity_from="Process", a_region_from="A",
            a_sector_to="Ore",      a_entity_to="Flow",      a_region_to="A",
            a_time="2020",          a_valuation="Mass",
            # B targets the CC valuation copy of the same physical pair
            b_sector_from="same",   b_entity_from="same",    b_region_from="same",
            b_sector_to="same",     b_entity_to="same",      b_region_to="same",
            b_time="same",          b_valuation="CC",
        )
        results = list(resolver.resolve_row(row))
        assert len(results) == 1
        assert results[0].a_edge_id == 0   # Mining→Ore/A 2020 Mass
        assert results[0].b_edge_id == 2   # Mining→Ore/A 2020 CC
