import pytest
import numpy as np
import pandas as pd
import os

from pysuto.acc import (
    EDGE_UNIVERSE_KEYS, AccountingStructure, EdgeBounds, _read_bounds_file,
)
from pysuto.masks import read_masks_csv


# =============================================================================
# THE BUILD HELPER
# =============================================================================

def _acc(root, *, id="t", forbid=(), same_region=False, bounds_file_path=None,
         masks_file_path=None, storage_dir="data/proc/acc"):
    """Build a structure through ``from_spec`` from a root dict (``nodes`` with
    sector / entity / region columns, ``time_range``, ``valuation`` = [gross, content]).

    Every sector pair is admissible (a ``use`` pair of every sector, all tradable) and
    every entity pair is listed in the accounting table, except the entity pairs in
    ``forbid``; ``same_region=True`` keeps every edge within one region.
    """
    nodes   = root["nodes"].drop_duplicates(["sector", "entity"])
    names   = list(nodes["sector"])
    entities = sorted(set(nodes["entity"]))
    gross, content = root["valuation"]
    spec = {
        "sectors":    [{"name": s, "entity": e}
                       for s, e in zip(nodes["sector"], nodes["entity"])],
        "regions":    list(dict.fromkeys(root["nodes"]["region"])),
        "valuations": [{"name": gross, "role": "gross"},
                       {"name": content, "role": "content"}],
        "times":      list(root["time_range"]),
        "supply":     {},
        "use":        {s: names for s in names},
        "tradable":   names,
        "accounting": [{"from": a, "to": b, "same_region": same_region}
                       for a in entities for b in entities if (a, b) not in forbid],
    }
    if bounds_file_path is not None and os.path.exists(bounds_file_path):
        spec["bounds"] = _read_bounds_file(bounds_file_path)
    if masks_file_path is not None:
        spec["masks"] = read_masks_csv(masks_file_path)
    return AccountingStructure.from_spec(spec, id, storage_dir=storage_dir)


# =============================================================================
# SHARED ROOT DEFINITION FIXTURE
# =============================================================================

@pytest.fixture
def root_simple():
    """Three nodes: Mining(P,A), Ore(F,A), Smelting(P,A)."""
    return {
        "nodes": pd.DataFrame({
            "sector": ["Mining",  "Ore",  "Smelting"],
            "entity": ["P",       "F",    "P"],
            "region": ["A",       "A",    "A"],
        }),
        "time_range": ["0"],
        "valuation":  ["mass", "copper_c"],
    }


@pytest.fixture
def root_multi_region():
    """Six nodes: Mining and Ore in both region A and B."""
    return {
        "nodes": pd.DataFrame({
            "sector": ["Mining", "Ore", "Mining", "Ore"],
            "entity": ["P",      "F",   "P",      "F"],
            "region": ["A",      "A",   "B",      "B"],
        }),
        "time_range": ["0", "1"],
        "valuation":  ["mass", "value"],
    }


@pytest.fixture
def root_multi_time():
    """Two nodes, two time steps, two valuations → 4 edges per node pair."""
    return {
        "nodes": pd.DataFrame({
            "sector": ["Mining", "Ore"],
            "entity": ["P",      "F"],
            "region": ["A",      "A"],
        }),
        "time_range": ["0", "1"],
        "valuation":  ["mass", "value"],
    }




# =============================================================================
# BOUNDS FIXTURES  (new)
# =============================================================================
 
@pytest.fixture
def root_bounds():
    """
    Four sectors: PSC, Mining, Smelting, WA.
    Two regions A and B, two times, two valuations.
    Designed so that bound rules can be applied at various
    sector/region/time/valuation granularities.
    """
    return {
        "nodes": pd.DataFrame({
            "sector": ["PSC",   "Mining", "Smelting", "WA",
                       "PSC",   "Mining", "Smelting", "WA"],
            "entity": ["P",     "P",      "F",        "P",
                       "P",     "P",      "F",        "P"],
            "region": ["A",     "A",      "A",        "A",
                       "B",     "B",      "B",        "B"],
        }),
        "time_range": ["2020", "2021"],
        "valuation":  ["mass", "copper_c"],
    }
 
 
@pytest.fixture
def acc_no_bounds(root_bounds):
    """AccountingStructure with no bounds file — all defaults."""
    return _acc(root=root_bounds)
 
 
@pytest.fixture
def bounds_df_psc():
    """
    Single rule: PSC sector_from=*, sector_to=*, valuation=* → l=-1e9, u=1e9.
    """
    return pd.DataFrame({
        "sector_from": ["PSC"],
        "sector_to":   ["*"],
        "valuation":   ["*"],
        "lower":       [-1e9],
        "upper":       [1e9],
    })
 
 
@pytest.fixture
def bounds_df_multi():
    """
    Three rules:
      1. PSC→*,*            l=-1e9, u=1e9
      2. PSC→*,mass         l=-1e6, u=1e6   (override row 1 for mass valuation)
      3. Mining→Smelting,copper_c,A,A  l=0, u=500
    """
    return pd.DataFrame({
        "sector_from": ["PSC",    "PSC",    "Mining"],
        "sector_to":   ["*",      "*",      "Smelting"],
        "valuation":   ["*",      "mass",   "copper_c"],
        "region_from": ["*",      "*",      "A"],
        "region_to":   ["*",      "*",      "A"],
        "lower":       [-1e9,     -1e6,     0.0],
        "upper":       [1e9,      1e6,      500.0],
    })
 
 
@pytest.fixture
def bounds_df_missing_cols():
    """Bounds DataFrame missing required 'lower' column — should raise."""
    return pd.DataFrame({
        "sector_from": ["PSC"],
        "sector_to":   ["*"],
        "valuation":   ["*"],
        "upper":       [1e9],
        # 'lower' intentionally absent
    })
 
 
@pytest.fixture
def acc_with_psc_bounds(root_bounds, bounds_df_psc, tmp_path):
    """
    AccountingStructure whose bounds are injected via a real Excel file
    on disk (exercises the full _load_bounds → read_excel path).
    """
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "bounds"
    ws.append(["sector_from", "sector_to", "valuation", "lower", "upper"])
    ws.append(["PSC", "*", "*", -1e9, 1e9])
    path = str(tmp_path / "bounds.xlsx")
    wb.save(path)
    return _acc(root=root_bounds, bounds_file_path=path)
 
 


# =============================================================================
# 1. INITIALISATION & NODE INDEX
# =============================================================================

class TestNodeIndex:

    def test_node_count(self, root_simple):
        acc = _acc(root=root_simple)
        assert acc.n_nodes == 3

    def test_node_to_id_is_bijection(self, root_simple):
        acc = _acc(root=root_simple)
        # node_to_id and id_to_node must be inverses of each other
        for key, nid in acc.node_to_id.items():
            assert acc.id_to_node[nid] == key

    def test_all_nodes_registered(self, root_simple):
        acc = _acc(root=root_simple)
        expected = {("Mining", "P", "A"), ("Ore", "F", "A"), ("Smelting", "P", "A")}
        assert set(acc.node_to_id.keys()) == expected

    def test_node_index_dataframe_shape(self, root_simple):
        acc = _acc(root=root_simple)
        assert acc.node_index.shape == (3, 4)
        assert list(acc.node_index.columns) == ["sector", "entity", "region", "entity_sub"]

# =============================================================================
# 2. EDGE INDEX — NO RULES
# =============================================================================

class TestEdgeIndexNoRules:

    def test_edge_count_no_rules(self, root_simple):
        """With no rules: n_nodes² × n_times × n_valuations edges."""
        acc = _acc(root=root_simple)
        expected = 3 * 3 * 1 * 2   # 9 node pairs × 1 time × 2 valuations
        assert len(acc.edge_index) == expected

    def test_edge_count_multi_time_valuation(self, root_multi_time):
        acc = _acc(root=root_multi_time)
        expected = 2 * 2 * 2 * 2   # 4 node pairs × 2 times × 2 valuations
        assert len(acc.edge_index) == expected

    def test_edge_ids_are_unique(self, root_simple):
        acc = _acc(root=root_simple)
        assert acc.edge_index["edge_id"].nunique() == len(acc.edge_index)

    def test_edge_ids_are_contiguous(self, root_simple):
        acc = _acc(root=root_simple)
        ids = sorted(acc.edge_index["edge_id"].tolist())
        assert ids == list(range(len(ids)))

    def test_edge_index_columns(self, root_simple):
        acc = _acc(root=root_simple)
        assert set(acc.edge_index.columns) == {"edge_id", "from_id", "to_id", "time", "valuation"}

    def test_all_node_pairs_present(self, root_simple):
        acc = _acc(root=root_simple)
        pairs = set(zip(acc.edge_index["from_id"], acc.edge_index["to_id"]))
        # 3 nodes → 9 pairs
        assert len(pairs) == 9

    def test_edge_index_all_geq_edge_index(self, root_simple):
        """edge_index_all must have at least as many rows as edge_index."""
        acc = _acc(root=root_simple, forbid=[("P", "P")])
        assert len(acc.edge_index_all) >= len(acc.edge_index)

    def test_n_edges_attribute_matches(self, root_simple):
        acc = _acc(root=root_simple)
        assert acc.n_edges == len(acc.edge_index)


# =============================================================================
# 3. ADMISSIBILITY MASK & RULES
# =============================================================================

class TestAdmissibilityRules:

    def test_no_p_to_p_removes_correct_edges(self, root_simple):
        """Mining(P)→Smelting(P) must be absent; Mining(P)→Ore(F) must be present."""
        acc = _acc(root=root_simple, forbid=[("P", "P")])

        mining_id   = acc.node_to_id[("Mining",  "P", "A")]
        ore_id      = acc.node_to_id[("Ore",     "F", "A")]
        smelting_id = acc.node_to_id[("Smelting","P", "A")]

        ei = acc.edge_index
        assert not ((ei["from_id"] == mining_id)  & (ei["to_id"] == smelting_id)).any()
        assert     ((ei["from_id"] == mining_id)  & (ei["to_id"] == ore_id)).any()

    def test_no_f_to_f_removes_correct_edges(self, root_simple):
        acc    = _acc(root=root_simple, forbid=[("F", "F")])
        ore_id = acc.node_to_id[("Ore", "F", "A")]
        ei     = acc.edge_index
        assert not ((ei["from_id"] == ore_id) & (ei["to_id"] == ore_id)).any()

    def test_no_cross_region_rule(self, root_multi_region):
        acc = _acc(root=root_multi_region, same_region=True)
        ei  = acc.edge_index
        node_region = {nid: t[2] for t, nid in acc.node_to_id.items()}
        from_regions = ei["from_id"].map(node_region)
        to_regions   = ei["to_id"].map(node_region)
        assert (from_regions == to_regions).all()

    def test_combined_rules_intersection(self, root_simple):
        acc_both = _acc(root=root_simple, forbid=[("P", "P"), ("F", "F")])
        acc_one  = _acc(root=root_simple,
                                       forbid=[("P", "P")])
        assert len(acc_both.edge_index) <= len(acc_one.edge_index)

    def test_always_false_rule_yields_zero_edges(self, root_simple):
        acc = _acc(root=root_simple, forbid=[("P", "P"), ("P", "F"), ("F", "P"), ("F", "F")])
        assert len(acc.edge_index) == 0

    def test_no_rules_yields_full_edge_set(self, root_simple):
        acc      = _acc(root=root_simple)
        expected = 3 * 3 * 1 * 2
        assert len(acc.edge_index) == expected

    def test_A_nodes_shape(self, root_simple):
        acc = _acc(root=root_simple, forbid=[("P", "P")])
        assert acc.A_nodes.shape == (3, 3)

    def test_A_tensor_shape(self, root_simple):
        """Full tensor must be 4-D: (n_nodes, n_nodes, n_timeps, n_valtns)."""
        acc = _acc(root=root_simple, forbid=[("P", "P")])
        assert acc.A.shape == (3, 3, 1, 2)

    def test_A_tensor_shape_multi(self, root_multi_time):
        acc = _acc(root=root_multi_time)
        assert acc.A.shape == (2, 2, 2, 2)

    def test_A_nodes_consistent_with_edge_index(self, root_simple):
        """Every (from_id, to_id) in edge_index must be 1 in A_nodes."""
        acc     = _acc(root=root_simple, forbid=[("P", "P")])
        A_dense = acc.A_nodes.todense()
        for _, row in acc.edge_index.iterrows():
            assert A_dense[int(row["from_id"]), int(row["to_id"])] == 1

    def test_A_tensor_consistent_with_edge_index(self, root_multi_time):
        """Every edge in edge_index must be 1 in the full tensor."""
        acc     = _acc(root=root_multi_time)
        in_A = set(zip(*acc.A.coords.tolist()))
        for _, row in acc.edge_index.iterrows():
            t_idx = acc.time_to_idx[row["time"]]
            v_idx = acc.val_to_idx[row["valuation"]]
            assert (int(row["from_id"]), int(row["to_id"]), t_idx, v_idx) in in_A

    def test_A_tensor_nnz_equals_n_edges(self, root_simple):
        """Number of nonzeros in A must equal number of admissible edges.
        root_simple has Mining(P), Ore(F), Smelting(P) — forbidding P→P blocks
        Mining→Smelting and Smelting→Mining, leaving 7 of 9 node pairs × 1 time
        × 2 valuations = 14 admissible edges, matching A.nnz exactly.
        """
        acc = _acc(root=root_simple, forbid=[("P", "P")])
        assert acc.A.nnz == acc.n_edges

# =============================================================================
# 4. TIME AND VALUATION DIMENSIONS
# =============================================================================

class TestTimeValuation:

    def test_all_time_steps_present(self, root_multi_time):
        acc = _acc(root=root_multi_time)
        assert set(acc.edge_index["time"].unique()) == {"0", "1"}

    def test_all_valuations_present(self, root_multi_time):
        acc = _acc(root=root_multi_time)
        assert set(acc.edge_index["valuation"].unique()) == {"mass", "value"}

    def test_edge_count_scales_with_time(self):
        """Doubling time steps must double edge count."""
        root_1t = {
            "nodes": pd.DataFrame({"sector": ["A","B"], "entity": ["P","F"], "region": ["X","X"]}),
            "time_range": ["0"], "valuation": ["mass", "copper_c"]
        }
        root_2t = {
            "nodes": pd.DataFrame({"sector": ["A","B"], "entity": ["P","F"], "region": ["X","X"]}),
            "time_range": ["0", "1"], "valuation": ["mass", "copper_c"]
        }
        acc_1 = _acc(root=root_1t)
        acc_2 = _acc(root=root_2t)
        assert len(acc_2.edge_index) == 2 * len(acc_1.edge_index)


# =============================================================================
# 7. SAVE / LOAD ROUND-TRIP
# =============================================================================

class TestSaveLoad:

    def _build(self, storage_dir):
        root = {
            "nodes": pd.DataFrame({
                "sector": ["Mining", "Ore", "Smelting"],
                "entity": ["P",      "F",   "P"],
                "region": ["A",      "A",   "A"],
            }),
            "time_range": ["0", "1"],
            "valuation":  ["mass", "value"],
        }
        return _acc(root, id="save_test", forbid=[("P", "P")], storage_dir=storage_dir)

    def test_save_load_node_to_id(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.node_to_id == acc.node_to_id

    def test_save_load_id_to_node(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.id_to_node == acc.id_to_node

    def test_save_load_edge_index_shape(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.edge_index.shape == acc.edge_index.shape

    def test_save_load_edge_index_values(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        expected = acc.edge_index.reset_index(drop=True).copy()
        expected["time"] = expected["time"].astype(str)
        pd.testing.assert_frame_equal(
            loaded.edge_index.reset_index(drop=True),
            expected,
            check_dtype=False,
        )

    def test_save_load_A_tensor_shape(self, tmp_path):
        """4-D tensor shape must survive round-trip."""
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.A.shape == acc.A.shape

    def test_save_load_A_tensor_nnz(self, tmp_path):
        """Non-zero count of 4-D tensor must survive round-trip."""
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.A.nnz == acc.A.nnz

    def test_save_load_A_nodes_shape(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.A_nodes.shape == acc.A_nodes.shape

    def test_save_load_time_valuation(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.time_range == acc.time_range
        assert loaded.valuation  == acc.valuation

    def test_save_load_dim_index_maps(self, tmp_path):
        """time_to_idx and val_to_idx must be reconstructed correctly on load."""
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.time_to_idx == acc.time_to_idx
        assert loaded.val_to_idx  == acc.val_to_idx

    def test_save_load_n_edges_all_attribute(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        assert loaded.n_edges_all == acc.n_edges_all

    def test_save_load_node_index(self, tmp_path):
        acc = self._build(str(tmp_path))
        acc.save()
        loaded = AccountingStructure.load("save_test", storage_dir=str(tmp_path))
        pd.testing.assert_frame_equal(
            loaded.node_index.reset_index(drop=True),
            acc.node_index.reset_index(drop=True),
            check_dtype=False,
        )

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scripts/acc.py --id nonexistent_id"):
            AccountingStructure.load("nonexistent_id", storage_dir=str(tmp_path))


# =============================================================================
# 11. EDGEBOUNDS — DEFAULTS
# =============================================================================
 
class TestEdgeBoundsDefaults:
    """No bounds file → all edges get l=0, u=+inf."""
 
    def test_bounds_attribute_exists(self, acc_no_bounds):
        assert hasattr(acc_no_bounds, "bounds")
        assert isinstance(acc_no_bounds.bounds, EdgeBounds)
 
    def test_l_shape(self, acc_no_bounds):
        assert acc_no_bounds.bounds.l.shape == (acc_no_bounds.n_edges,)
 
    def test_u_shape(self, acc_no_bounds):
        assert acc_no_bounds.bounds.u.shape == (acc_no_bounds.n_edges,)
 
    def test_l_dtype(self, acc_no_bounds):
        assert acc_no_bounds.bounds.l.dtype == np.float64
 
    def test_u_dtype(self, acc_no_bounds):
        assert acc_no_bounds.bounds.u.dtype == np.float64
 
    def test_default_lower_is_zero(self, acc_no_bounds):
        assert np.all(acc_no_bounds.bounds.l == 0.0)
 
    def test_default_upper_is_the_practical_large_value(self, acc_no_bounds):
        assert np.all(acc_no_bounds.bounds.u == 1e9)
 
    def test_default_upper_is_positive(self, acc_no_bounds):
        assert np.all(acc_no_bounds.bounds.u > 0)
 
    def test_n_exceptions_is_zero_for_defaults(self, acc_no_bounds):
        assert acc_no_bounds.bounds.n_exceptions == 0
 
    def test_property_l_delegates_to_bounds(self, acc_no_bounds):
        """acc.l must be the same object as acc.bounds.l."""
        assert acc_no_bounds.l is acc_no_bounds.bounds.l
 
    def test_property_u_delegates_to_bounds(self, acc_no_bounds):
        """acc.u must be the same object as acc.bounds.u."""
        assert acc_no_bounds.u is acc_no_bounds.bounds.u
 
    def test_l_length_equals_n_edges(self, acc_no_bounds):
        assert len(acc_no_bounds.l) == acc_no_bounds.n_edges
 
    def test_u_length_equals_n_edges(self, acc_no_bounds):
        assert len(acc_no_bounds.u) == acc_no_bounds.n_edges
 
 
# =============================================================================
# 12. EDGEBOUNDS — RULE MATCHING
# =============================================================================
 
class TestEdgeBoundsRuleMatching:
    """EdgeBounds._apply_rules: correct edges receive the exception values."""
 
    def _make_bounds(self, root_bounds, bounds_df):
        """Build an EdgeBounds directly from a DataFrame (no Excel file needed)."""
        acc = _acc(root=root_bounds)
        return EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=bounds_df,
        ), acc
 
    def test_psc_rule_sets_negative_lower(self, root_bounds, bounds_df_psc):
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = [nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC"]
        psc_from_mask = acc.edge_index["from_id"].isin(psc_ids).values
        # every PSC-from edge must have l = -1e9
        assert np.all(eb.l[psc_from_mask] == -1e9)
 
    def test_psc_rule_sets_upper(self, root_bounds, bounds_df_psc):
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = [nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC"]
        psc_from_mask = acc.edge_index["from_id"].isin(psc_ids).values
        assert np.all(eb.u[psc_from_mask] == 1e9)
 
    def test_non_psc_edges_keep_default_lower(self, root_bounds, bounds_df_psc):
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc_mask = ~acc.edge_index["from_id"].isin(psc_ids).values
        assert np.all(eb.l[non_psc_mask] == 0.0)
 
    def test_non_psc_edges_keep_default_upper(self, root_bounds, bounds_df_psc):
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc_mask = ~acc.edge_index["from_id"].isin(psc_ids).values
        assert np.all(eb.u[non_psc_mask] == 1e9)
 
    def test_wildcard_sector_to_matches_all_destinations(self, root_bounds, bounds_df_psc):
        """sector_to='*' must apply regardless of destination sector."""
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        psc_edges = acc.edge_index[acc.edge_index["from_id"].isin(psc_ids)]
        dest_sectors = set(
            acc.decode_node(int(nid))[0]
            for nid in psc_edges["to_id"].unique()
        )
        # PSC should have edges to multiple destination sectors
        assert len(dest_sectors) > 1
        # all of them have l=-1e9 regardless of destination
        assert np.all(eb.l[psc_edges.index] == -1e9)
 
    def test_wildcard_valuation_matches_all_valuations(self, root_bounds, bounds_df_psc):
        """valuation='*' must apply to both mass and copper_c edges."""
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        psc_edges = acc.edge_index[acc.edge_index["from_id"].isin(psc_ids)]
        assert set(psc_edges["valuation"].unique()) == {"mass", "copper_c"}
        assert np.all(eb.l[psc_edges.index] == -1e9)
 
    def test_later_rule_overrides_earlier(self, root_bounds, bounds_df_multi):
        """
        Row 1: PSC→*, *  → l=-1e9
        Row 2: PSC→*, mass → l=-1e6   (must override row 1 for mass-valuation edges)
        """
        eb, acc = self._make_bounds(root_bounds, bounds_df_multi)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        psc_mass = acc.edge_index[
            acc.edge_index["from_id"].isin(psc_ids) &
            (acc.edge_index["valuation"] == "mass")
        ]
        psc_copper = acc.edge_index[
            acc.edge_index["from_id"].isin(psc_ids) &
            (acc.edge_index["valuation"] == "copper_c")
        ]
        # mass → overridden to -1e6
        assert np.all(eb.l[psc_mass.index] == -1e6)
        assert np.all(eb.u[psc_mass.index] == 1e6)
        # copper_c → only row 1 applies → -1e9
        assert np.all(eb.l[psc_copper.index] == -1e9)
        assert np.all(eb.u[psc_copper.index] == 1e9)
 
    def test_region_filter_restricts_match(self, root_bounds, bounds_df_multi):
        """
        Row 3: Mining→Smelting, copper_c, region_from=A, region_to=A → u=500
        Edges with region_from=B must keep default upper bound.
        """
        eb, acc = self._make_bounds(root_bounds, bounds_df_multi)
        mining_a = acc.node_to_id[("Mining", "P", "A")]
        mining_b = acc.node_to_id[("Mining", "P", "B")]
        smelting_a = acc.node_to_id[("Smelting", "F", "A")]
        smelting_b = acc.node_to_id[("Smelting", "F", "B")]
 
        ei = acc.edge_index
        # A→A copper_c
        aa = ei[
            (ei["from_id"] == mining_a) &
            (ei["to_id"]   == smelting_a) &
            (ei["valuation"] == "copper_c")
        ]
        assert len(aa) > 0
        assert np.all(eb.u[aa.index] == 500.0)
 
        # B→B copper_c — region filter excludes this
        bb = ei[
            (ei["from_id"] == mining_b) &
            (ei["to_id"]   == smelting_b) &
            (ei["valuation"] == "copper_c")
        ]
        assert len(bb) > 0
        assert np.all(eb.u[bb.index] == 1e9)
 
    def test_n_exceptions_counts_only_non_default(self, root_bounds, bounds_df_psc):
        eb, acc = self._make_bounds(root_bounds, bounds_df_psc)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        expected = acc.edge_index["from_id"].isin(psc_ids).sum()
        assert eb.n_exceptions == expected
 
    def test_a_rule_matching_no_edge_raises_naming_it(self, root_bounds):
        """A misspelt selector must not drop its bound silently."""
        bounds_df = pd.DataFrame({
            "sector_from": ["NONEXISTENT"],
            "sector_to":   ["*"],
            "valuation":   ["*"],
            "lower":       [-999.0],
            "upper":       [999.0],
        })
        acc = _acc(root=root_bounds)
        with pytest.raises(ValueError, match="NONEXISTENT.*matches no edge"):
            EdgeBounds(edge_index=acc.edge_index, node_index=acc.node_index,
                       bounds_df=bounds_df)

    def test_lower_above_upper_raises(self, root_bounds):
        bounds_df = pd.DataFrame({
            "sector_from": ["PSC"], "sector_to": ["*"], "valuation": ["*"],
            "lower":       [5.0],   "upper":     [1.0],
        })
        acc = _acc(root=root_bounds)
        with pytest.raises(ValueError, match="lower bound above their upper bound"):
            EdgeBounds(edge_index=acc.edge_index, node_index=acc.node_index,
                       bounds_df=bounds_df)

    def test_a_none_cell_is_a_wildcard(self, root_bounds):
        """A blank ODS cell reaches the rule as None: it selects everything."""
        bounds_df = pd.DataFrame({
            "sector_from": ["PSC"], "sector_to": [None], "valuation": [None],
            "lower":       [-1.0],  "upper":     [9.0],
        })
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(edge_index=acc.edge_index, node_index=acc.node_index,
                        bounds_df=bounds_df)
        assert eb.n_exceptions > 0

    def test_a_missing_bounds_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Bounds file not found"):
            AccountingStructure(id="t", root_file_path="root.ods",
                                rules_file_path="rules.ods",
                                bounds_file_path=str(tmp_path / "nope.ods"))

    def test_blank_lower_leaves_existing_lower_unchanged(self, root_bounds):
        """A rule with lower=NaN must not override the lower bound."""
        bounds_df = pd.DataFrame({
            "sector_from": ["PSC"],
            "sector_to":   ["*"],
            "valuation":   ["*"],
            "lower":       [np.nan],   # blank — should not change lower
            "upper":       [999.0],
        })
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=bounds_df,
        )
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        psc_mask = acc.edge_index["from_id"].isin(psc_ids).values
        # lower must remain 0 (default), upper must be 999
        assert np.all(eb.l[psc_mask] == 0.0)
        assert np.all(eb.u[psc_mask] == 999.0)
 
    def test_missing_required_column_raises(self, root_bounds, bounds_df_missing_cols):
        acc = _acc(root=root_bounds)
        with pytest.raises(ValueError, match="missing required columns"):
            EdgeBounds(
                edge_index=acc.edge_index,
                node_index=acc.node_index,
                bounds_df=bounds_df_missing_cols,
            )
 
    def test_empty_bounds_df_gives_all_defaults(self, root_bounds):
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=pd.DataFrame(
                columns=["sector_from", "sector_to", "valuation", "lower", "upper"]
            ),
        )
        assert np.all(eb.l == 0.0)
        assert np.all(eb.u == 1e9)


# =============================================================================
# 13. EDGEBOUNDS — VALIDATE
# =============================================================================
 
class TestEdgeBoundsValidate:
    """EdgeBounds.validate() returns empty DataFrame on compliance,
    reports violations on failure, and raises when requested."""
 
    @pytest.fixture
    def eb_with_psc(self, root_bounds, bounds_df_psc):
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=bounds_df_psc,
        )
        return eb, acc
 
    def test_validate_all_zeros_passes(self, eb_with_psc):
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.empty
 
    def test_validate_all_positive_passes_default_edges(self, eb_with_psc):
        eb, acc = eb_with_psc
        F = np.ones(acc.n_edges) * 100.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.empty
 
    def test_validate_negative_psc_passes(self, eb_with_psc):
        """Negative values on PSC edges are within their exception bounds."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        psc_mask = acc.edge_index["from_id"].isin(psc_ids).values
        F[psc_mask] = -500.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.empty
 
    def test_validate_negative_non_psc_fails(self, eb_with_psc):
        """Negative value on a non-PSC edge violates l=0."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc_mask = ~acc.edge_index["from_id"].isin(psc_ids).values
        first_non_psc = np.where(non_psc_mask)[0][0]
        F[first_non_psc] = -1.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert len(result) == 1
        assert result.iloc[0]["violation"] == "below_lower"
 
    def test_validate_above_upper_fails(self, eb_with_psc):
        """Value exceeding u=1e9 on a PSC edge must be reported."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        first_psc = np.where(acc.edge_index["from_id"].isin(psc_ids).values)[0][0]
        F[first_psc] = 2e9   # exceeds u=1e9
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert len(result) >= 1
        assert "above_upper" in result["violation"].values
 
    def test_validate_multiple_violations_all_reported(self, eb_with_psc):
        """All violating edges must appear in the report, not just the first."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        # Force violations on first three non-PSC edges
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc_indices = np.where(~acc.edge_index["from_id"].isin(psc_ids).values)[0]
        for idx in non_psc_indices[:3]:
            F[idx] = -1.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert len(result) == 3
 
    def test_validate_raises_by_default(self, eb_with_psc):
        """raise_on_violation=True (default) must raise ValueError."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc = np.where(~acc.edge_index["from_id"].isin(psc_ids).values)[0][0]
        F[non_psc] = -1.0
        with pytest.raises(ValueError):
            eb.validate(F, acc.edge_index)
 
    def test_validate_does_not_raise_when_flag_false(self, eb_with_psc):
        """raise_on_violation=False must not raise even when violations exist."""
        eb, acc = eb_with_psc
        F = np.full(acc.n_edges, -1.0)   # all below lower=0 for non-PSC
        try:
            eb.validate(F, acc.edge_index, raise_on_violation=False)
        except ValueError:
            pytest.fail("validate() raised despite raise_on_violation=False")
 
    def test_validate_report_columns(self, eb_with_psc):
        """Violation report must contain the documented columns."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc = np.where(~acc.edge_index["from_id"].isin(psc_ids).values)[0][0]
        F[non_psc] = -1.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        required_cols = {"edge_id", "from_id", "to_id", "time", "valuation",
                         "lower", "value", "upper", "violation"}
        assert required_cols.issubset(set(result.columns))
 
    def test_validate_report_value_column_matches_input(self, eb_with_psc):
        """The 'value' column in the report must equal the input F at that edge."""
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc = np.where(~acc.edge_index["from_id"].isin(psc_ids).values)[0][0]
        F[non_psc] = -42.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.iloc[0]["value"] == pytest.approx(-42.0)
 
    def test_validate_below_lower_label(self, eb_with_psc):
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc = np.where(~acc.edge_index["from_id"].isin(psc_ids).values)[0][0]
        F[non_psc] = -1.0
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.iloc[0]["violation"] == "below_lower"
 
    def test_validate_above_upper_label(self, eb_with_psc):
        eb, acc = eb_with_psc
        F = np.zeros(acc.n_edges)
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        first_psc = np.where(acc.edge_index["from_id"].isin(psc_ids).values)[0][0]
        F[first_psc] = 2e9
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        psc_viol = result[result["violation"] == "above_upper"]
        assert len(psc_viol) >= 1
 
    def test_validate_exactly_at_lower_bound_passes(self, eb_with_psc):
        """F = l exactly must not be flagged as a violation."""
        eb, acc = eb_with_psc
        F = eb.l.copy()   # set every edge exactly to its lower bound
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.empty
 
    def test_validate_exactly_at_upper_bound_passes(self, eb_with_psc):
        """F = u exactly must not be flagged (finite upper bounds only)."""
        eb, acc = eb_with_psc
        F = eb.l.copy()
        finite_mask = np.isfinite(eb.u)
        F[finite_mask] = eb.u[finite_mask]
        result = eb.validate(F, acc.edge_index, raise_on_violation=False)
        assert result.empty
 
 
# =============================================================================
# 14. EDGEBOUNDS — ACCOUNTING STRUCTURE INTEGRATION
# =============================================================================
 
class TestEdgeBoundsAccIntegration:
    """Test bounds loaded via bounds_file_path= on AccountingStructure."""
 
    def test_acc_bounds_attribute_is_edge_bounds(self, acc_with_psc_bounds):
        assert isinstance(acc_with_psc_bounds.bounds, EdgeBounds)
 
    def test_acc_l_property_shape(self, acc_with_psc_bounds):
        assert acc_with_psc_bounds.l.shape == (acc_with_psc_bounds.n_edges,)
 
    def test_acc_u_property_shape(self, acc_with_psc_bounds):
        assert acc_with_psc_bounds.u.shape == (acc_with_psc_bounds.n_edges,)
 
    def test_psc_edges_have_negative_lower_via_file(self, acc_with_psc_bounds):
        acc = acc_with_psc_bounds
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        psc_mask = acc.edge_index["from_id"].isin(psc_ids).values
        assert np.all(acc.l[psc_mask] == -1e9)
 
    def test_non_psc_edges_have_default_lower_via_file(self, acc_with_psc_bounds):
        acc = acc_with_psc_bounds
        psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
        non_psc_mask = ~acc.edge_index["from_id"].isin(psc_ids).values
        assert np.all(acc.l[non_psc_mask] == 0.0)
 
    def test_no_bounds_file_kwarg_gives_defaults(self, root_bounds):
        acc = _acc(root=root_bounds)
        assert np.all(acc.l == 0.0)
        assert np.all(acc.u == 1e9)
 
 
# =============================================================================
# 15. EDGEBOUNDS — SAVE / LOAD ROUND-TRIP
# =============================================================================
 
class TestEdgeBoundsPersistence:
    """Bounds must survive save()/load() bit-for-bit."""
 
    def _build_with_bounds(self, storage_dir, root_bounds, bounds_df_psc, tmp_path):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "bounds"
        ws.append(["sector_from", "sector_to", "valuation", "lower", "upper"])
        ws.append(["PSC", "*", "*", -1e9, 1e9])
        path = str(tmp_path / "bounds.xlsx")
        wb.save(path)
        return _acc(root_bounds, id="bounds_save_test", bounds_file_path=path,
                    storage_dir=storage_dir)
 
    def test_l_survives_round_trip(self, tmp_path, root_bounds, bounds_df_psc):
        acc = self._build_with_bounds(str(tmp_path), root_bounds, bounds_df_psc, tmp_path)
        acc.save()
        loaded = AccountingStructure.load("bounds_save_test", storage_dir=str(tmp_path))
        np.testing.assert_array_equal(loaded.l, acc.l)
 
    def test_u_survives_round_trip(self, tmp_path, root_bounds, bounds_df_psc):
        acc = self._build_with_bounds(str(tmp_path), root_bounds, bounds_df_psc, tmp_path)
        acc.save()
        loaded = AccountingStructure.load("bounds_save_test", storage_dir=str(tmp_path))
        np.testing.assert_array_equal(loaded.u, acc.u)
 
    def test_n_exceptions_survives_round_trip(self, tmp_path, root_bounds, bounds_df_psc):
        acc = self._build_with_bounds(str(tmp_path), root_bounds, bounds_df_psc, tmp_path)
        acc.save()
        loaded = AccountingStructure.load("bounds_save_test", storage_dir=str(tmp_path))
        assert loaded.bounds.n_exceptions == acc.bounds.n_exceptions
 
    def test_load_without_npy_files_gives_defaults(self, tmp_path, root_bounds):
        """If .npy files are absent (legacy save), load falls back to defaults."""
        acc = _acc(root_bounds, id="legacy_test", storage_dir=str(tmp_path))
        acc.save()
        # Remove the bounds .npy files to simulate a legacy save
        os.remove(tmp_path / "legacy_test" / "bounds_l.npy")
        os.remove(tmp_path / "legacy_test" / "bounds_u.npy")
        loaded = AccountingStructure.load("legacy_test", storage_dir=str(tmp_path))
        assert np.all(loaded.l == 0.0)
        assert np.all(loaded.u == 1e9)
 
    def test_npy_files_exist_after_save(self, tmp_path, root_bounds):
        acc = _acc(root_bounds, id="npy_test", storage_dir=str(tmp_path))
        acc.save()
        assert (tmp_path / "npy_test" / "bounds_l.npy").exists()
        assert (tmp_path / "npy_test" / "bounds_u.npy").exists()
 
    def test_loaded_bounds_pass_validate_on_compliant_F(
        self, tmp_path, root_bounds, bounds_df_psc
    ):
        """validate() on a loaded structure must behave identically to the original."""
        acc = self._build_with_bounds(str(tmp_path), root_bounds, bounds_df_psc, tmp_path)
        acc.save()
        loaded = AccountingStructure.load("bounds_save_test", storage_dir=str(tmp_path))
        F = np.zeros(loaded.n_edges)
        result = loaded.bounds.validate(F, loaded.edge_index, raise_on_violation=False)
        assert result.empty
 
 
class TestEdgeBoundsInvariants:
    """Invariants that must hold regardless of rules or data."""
 
    def test_l_never_exceeds_u(self, root_bounds, bounds_df_multi):
        """l ≤ u must hold for every edge after rule application."""
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=bounds_df_multi,
        )
        assert np.all(eb.l <= eb.u)
 
    def test_bounds_aligned_to_edge_index_length(self, root_bounds, bounds_df_psc):
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=bounds_df_psc,
        )
        assert len(eb.l) == len(acc.edge_index)
        assert len(eb.u) == len(acc.edge_index)
 
    def test_summary_returns_string(self, acc_no_bounds):
        s = acc_no_bounds.bounds.summary()
        assert isinstance(s, str)
        assert "EdgeBounds" in s
 
    def test_bounds_independent_of_admissibility_rules(self, root_bounds, bounds_df_psc):
        """
        Applying admissibility rules changes n_edges but must not change
        which edges have their bounds set — the PSC rule still applies to
        whatever PSC edges survive.
        """
        acc_all  = _acc(root=root_bounds)
        acc_same = _acc(root=root_bounds, same_region=True)
 
        eb_all  = EdgeBounds(acc_all.edge_index,  acc_all.node_index,  bounds_df_psc)
        eb_same = EdgeBounds(acc_same.edge_index, acc_same.node_index, bounds_df_psc)
 
        # In both cases all PSC-from edges must have l=-1e9
        for acc, eb in [(acc_all, eb_all), (acc_same, eb_same)]:
            psc_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "PSC")
            mask = acc.edge_index["from_id"].isin(psc_ids).values
            assert np.all(eb.l[mask] == -1e9), (
                f"PSC lower bound not set for acc with {acc.n_edges} edges"
            )
 
    def test_time_filter_applies_correctly(self, root_bounds):
        """A rule with time='2020' must only affect edges at t=2020."""
        bounds_df = pd.DataFrame({
            "sector_from": ["Mining"],
            "sector_to":   ["*"],
            "valuation":   ["*"],
            "time":        ["2020"],
            "lower":       [0.0],
            "upper":       [100.0],
        })
        acc = _acc(root=root_bounds)
        eb = EdgeBounds(
            edge_index=acc.edge_index,
            node_index=acc.node_index,
            bounds_df=bounds_df,
        )
        mining_ids = set(nid for (s, e, r), nid in acc.node_to_id.items() if s == "Mining")
        ei = acc.edge_index
 
        # t=2020, Mining-from: upper must be 100
        t2020_mining = ei[
            ei["from_id"].isin(mining_ids) & (ei["time"] == "2020")
        ]
        assert np.all(eb.u[t2020_mining.index] == 100.0)
 
        # t=2021, Mining-from: upper must be default (1e9)
        t2021_mining = ei[
            ei["from_id"].isin(mining_ids) & (ei["time"] == "2021")
        ]
        assert np.all(eb.u[t2021_mining.index] == 1e9)
 
 
class TestCrossRegionMask:
    """Built-in structural mask ``cross_region`` (mapped self-trade scope, ADR 0014)."""

    def test_cross_region_builtin(self, root_multi_region):
        acc = _acc(root=root_multi_region)
        assert "cross_region" in acc.masks
        ei = acc.edge_index
        rf = acc.node_index.loc[ei["from_id"].values, "region"].to_numpy()
        rt = acc.node_index.loc[ei["to_id"].values, "region"].to_numpy()
        np.testing.assert_array_equal(acc.masks["cross_region"], rf != rt)
        # the two-region toy has both trade and domestic edges
        assert acc.masks["cross_region"].any()
        assert (~acc.masks["cross_region"]).any()



# =============================================================================
# USER MASKS SURVIVE SAVE → LOAD (ADR 0038)
# =============================================================================

def _write_masks(path):
    path.write_text("mask_name,logic,sector_from,sector_to\n"
                    "scope,include,*,*\n"
                    "scope,exclude,Mining,Ore\n")


def test_user_masks_survive_save_and_load(root_simple, tmp_path):
    """masks.csv is read only when the structure is built; load() must restore
    the masks from the saved structure, else the Stage-3 mass-balance scope
    silently vanishes."""
    csv = tmp_path / "masks.csv"
    _write_masks(csv)
    store = str(tmp_path / "proc")
    acc = _acc(root=root_simple, masks_file_path=str(csv),
                              storage_dir=store)
    assert acc.user_mask_names == ["scope"]
    assert (~acc.masks["scope"]).any()
    acc.save()
    loaded = AccountingStructure.load("t", storage_dir=store)
    assert loaded.user_mask_names == ["scope"]
    np.testing.assert_array_equal(loaded.masks["scope"], acc.masks["scope"])


def test_save_without_user_masks_drops_the_stale_file(root_simple, tmp_path):
    csv = tmp_path / "masks.csv"
    _write_masks(csv)
    store = str(tmp_path / "proc")
    _acc(root=root_simple, masks_file_path=str(csv),
                        storage_dir=store).save()
    _acc(root=root_simple, storage_dir=store).save()
    loaded = AccountingStructure.load("t", storage_dir=store)
    assert "scope" not in loaded.masks
    assert loaded.user_mask_names == []


# =============================================================================
# THE BUILD RECORD: WHICH ACC INPUT VERSION A BUILT STRUCTURE CAME FROM (#53)
# =============================================================================

class TestBuildRecord:
    """A built structure keeps a build record of the acc input version it was built
    from; a stage that loads the structure checks the record against its acc ref."""

    ENTRY = {"id": "t", "version": 4, "date": "2026-09-29T10:53:03Z",
             "code": "dd2e5794", "label": "t"}

    def _built(self, root_simple, tmp_path):
        store = str(tmp_path / "proc" / "acc")
        _acc(root=root_simple, storage_dir=store).save()
        return store

    @staticmethod
    def _inputs(tmp_path, specs=None):
        """The acc input registry the record's versions are read from: t v1 … v4, each
        with its own regions (another edge universe) unless *specs* says otherwise."""
        from test_acc_roles import COPPER_LABELS, acc_input_versions
        if specs is None:
            specs = [dict(COPPER_LABELS, regions=[f"R{i}" for i in range(n)])
                     for n in (1, 2, 3, 4)]
        return acc_input_versions(tmp_path / "input", "t", specs)

    def test_the_record_holds_the_entrys_id_version_and_code(self, root_simple, tmp_path):
        import json
        from pysuto.acc import BUILD_RECORD, write_build_record
        store = self._built(root_simple, tmp_path)
        path = write_build_record(store, self.ENTRY)
        assert path == os.path.join(store, "t", BUILD_RECORD)
        assert BUILD_RECORD == "build_record.json"
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"id": "t", "version": 4, "code": "dd2e5794"}

    def test_a_matching_ref_passes_silently(self, root_simple, tmp_path, caplog):
        import logging
        from pysuto.acc import check_built_acc, write_build_record
        store = self._built(root_simple, tmp_path)
        write_build_record(store, self.ENTRY)
        reg = self._inputs(tmp_path)
        with caplog.at_level(logging.WARNING):
            check_built_acc(store, {"id": "t", "version": 4}, reg, EDGE_UNIVERSE_KEYS)
            check_built_acc(store, {"id": "t", "version": 4}, reg, EDGE_UNIVERSE_KEYS,
                            require_record=True)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_another_edge_universe_raises_naming_both_versions_and_the_key(
            self, root_simple, tmp_path):
        from pysuto.acc import check_built_acc, write_build_record
        store = self._built(root_simple, tmp_path)
        write_build_record(store, self.ENTRY)
        with pytest.raises(ValueError, match=r"t v4.*t v3; they differ in \['regions'\]"):
            check_built_acc(store, {"id": "t", "version": 3, "code": "0342a990"},
                            self._inputs(tmp_path), EDGE_UNIVERSE_KEYS)

    def test_another_version_with_the_same_edge_universe_passes(self, root_simple,
                                                                tmp_path, caplog):
        """#119: bounds and masks changes pass Stages 4 and 6 (info only), a role change
        raises there, and Stages 2 and 3, whose G and recipes hold all of them, refuse
        each."""
        import logging
        from pysuto.acc import (ALL_SPEC_KEYS, EDGE_UNIVERSE_ROLE_KEYS, check_built_acc,
                                write_build_record)
        from test_acc_roles import BOUNDS_ONLY, COPPER_LABELS, MASKS_ONLY, ROLE_ONLY
        store = self._built(root_simple, tmp_path)
        write_build_record(store, self.ENTRY)
        reg = self._inputs(tmp_path, [MASKS_ONLY, BOUNDS_ONLY, ROLE_ONLY, COPPER_LABELS])
        keys = EDGE_UNIVERSE_ROLE_KEYS
        with caplog.at_level(logging.INFO):
            for v in (1, 2):
                check_built_acc(store, {"id": "t", "version": v}, reg, keys)
        assert "differ in ['masks']" in caplog.text
        assert "differ in ['bounds']" in caplog.text
        for v, key in ((1, "masks"), (2, "bounds"), (3, "role")):
            with pytest.raises(ValueError, match=rf"they differ in \['{key}'\]"):
                check_built_acc(store, {"id": "t", "version": v}, reg, ALL_SPEC_KEYS)
        with pytest.raises(ValueError, match=r"they differ in \['role'\]"):
            check_built_acc(store, {"id": "t", "version": 3}, reg, keys)

    def test_a_structure_without_the_record_warns_and_continues(self, root_simple,
                                                                tmp_path, caplog):
        """The copper structure was built before the record existed; the final chain
        must still render."""
        import logging
        from pysuto.acc import check_built_acc
        store = self._built(root_simple, tmp_path)
        with caplog.at_level(logging.WARNING):
            check_built_acc(store, {"id": "t", "version": 4}, self._inputs(tmp_path),
                            EDGE_UNIVERSE_KEYS)
        assert "no build record" in caplog.text and "t v4" in caplog.text

    def test_a_structure_without_the_record_raises_when_one_is_required(self, root_simple,
                                                                        tmp_path):
        """A pinned run cannot verify its acc version against an unrecorded build."""
        from pysuto.acc import check_built_acc
        store = self._built(root_simple, tmp_path)
        with pytest.raises(ValueError, match="no build record"):
            check_built_acc(store, {"id": "t", "version": 4}, self._inputs(tmp_path),
                            EDGE_UNIVERSE_KEYS, require_record=True)

    @pytest.mark.parametrize("ref", [{}, {"id": "t"}, {"version": 4},
                                     {"id": "", "version": 4}, {"id": "t", "version": None}])
    def test_an_empty_or_partial_acc_ref_raises(self, root_simple, tmp_path, ref):
        """Nothing to check against is a defect of the caller's sidecar, never a pass."""
        from pysuto.acc import check_built_acc, write_build_record
        store = self._built(root_simple, tmp_path)
        write_build_record(store, self.ENTRY)
        with pytest.raises(ValueError, match="acc ref"):
            check_built_acc(store, ref, self._inputs(tmp_path), EDGE_UNIVERSE_KEYS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])