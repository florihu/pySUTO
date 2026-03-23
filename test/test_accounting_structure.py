import pytest
import numpy as np
import pandas as pd
import sparse
import os

from pysuto.AccountingStructure import AccountingStructure


# =============================================================================
# SHARED RULES (reusable across tests)
# =============================================================================

def rule_no_p_to_p(ni, nj):
    """Producers cannot transact directly with other producers."""
    return not (ni["entity"] == "P" and nj["entity"] == "P")

def rule_no_f_to_f(ni, nj):
    """Flows cannot go directly between two flow nodes."""
    return not (ni["entity"] == "F" and nj["entity"] == "F")

def rule_no_self_loop(ni, nj):
    """A node cannot have an edge to itself."""
    return not (ni["sector"] == nj["sector"] and
                ni["entity"] == nj["entity"] and
                ni["region"] == nj["region"])

def rule_no_cross_region(ni, nj):
    """Edges cannot cross regions."""
    return ni["region"] == nj["region"]

def rule_always_false(ni, nj):
    """Blocks all edges — useful to test zero-edge structures."""
    return False

# Edge-level rules (4 positional args: ni, nj, time, valuation)

def edge_rule_only_time_0(ni, nj, time, valuation):
    """Only allow edges in time step 0."""
    return time == 0

def edge_rule_only_mass(ni, nj, time, valuation):
    """Only allow edges with valuation='mass'."""
    return valuation == "mass"

def edge_rule_always_false(ni, nj, time, valuation):
    """Blocks all edges at the edge level."""
    return False


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
        "time_range": [0],
        "valuation":  ["mass"],
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
        "time_range": [0, 1],
        "valuation":  ["mass", "value"],
    }


@pytest.fixture
def root_with_duplicates():
    """time_range and valuation contain duplicates — must be silently deduped."""
    return {
        "nodes": pd.DataFrame({
            "sector": ["Mining", "Ore"],
            "entity": ["P",      "F"],
            "region": ["A",      "A"],
        }),
        "time_range": [0, 1, 0],            # duplicate 0
        "valuation":  ["mass", "mass"],     # duplicate mass
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
        "time_range": [0, 1],
        "valuation":  ["mass", "value"],
    }


# =============================================================================
# 1. INITIALISATION & NODE INDEX
# =============================================================================

class TestNodeIndex:

    def test_node_count(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        assert acc.n_nodes == 3

    def test_node_to_id_is_bijection(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        # node_to_id and id_to_node must be inverses of each other
        for key, nid in acc.node_to_id.items():
            assert acc.id_to_node[nid] == key

    def test_all_nodes_registered(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        expected = {("Mining", "P", "A"), ("Ore", "F", "A"), ("Smelting", "P", "A")}
        assert set(acc.node_to_id.keys()) == expected

    def test_duplicate_nodes_deduplicated(self):
        """Duplicate rows in root nodes must not create duplicate IDs."""
        root = {
            "nodes": pd.DataFrame({
                "sector": ["Mining", "Mining"],
                "entity": ["P",      "P"],
                "region": ["A",      "A"],
            }),
            "time_range": [0],
            "valuation":  ["mass"],
        }
        acc = AccountingStructure(id="t", root=root)
        assert acc.n_nodes == 1

    def test_node_index_dataframe_shape(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        assert acc.node_index.shape == (3, 3)
        assert list(acc.node_index.columns) == ["sector", "entity", "region"]

    def test_encode_decode_roundtrip(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        nid = acc.encode_node("Mining", "P", "A")
        assert acc.decode_node(nid) == ("Mining", "P", "A")


# =============================================================================
# 2. EDGE INDEX — NO RULES
# =============================================================================

class TestEdgeIndexNoRules:

    def test_edge_count_no_rules(self, root_simple):
        """With no rules: n_nodes² × n_times × n_valuations edges."""
        acc = AccountingStructure(id="t", root=root_simple)
        expected = 3 * 3 * 1 * 1   # 9 node pairs × 1 time × 1 valuation
        assert len(acc.edge_index) == expected

    def test_edge_count_multi_time_valuation(self, root_multi_time):
        acc = AccountingStructure(id="t", root=root_multi_time)
        expected = 2 * 2 * 2 * 2   # 4 node pairs × 2 times × 2 valuations
        assert len(acc.edge_index) == expected

    def test_edge_ids_are_unique(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        assert acc.edge_index["edge_id"].nunique() == len(acc.edge_index)

    def test_edge_ids_are_contiguous(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        ids = sorted(acc.edge_index["edge_id"].tolist())
        assert ids == list(range(len(ids)))

    def test_edge_index_columns(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        assert set(acc.edge_index.columns) == {"edge_id", "from_id", "to_id", "time", "valuation"}

    def test_all_node_pairs_present(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        pairs = set(zip(acc.edge_index["from_id"], acc.edge_index["to_id"]))
        # 3 nodes → 9 pairs
        assert len(pairs) == 9

    def test_edge_index_all_geq_edge_index(self, root_simple):
        """edge_index_all must have at least as many rows as edge_index."""
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        assert len(acc.edge_index_all) >= len(acc.edge_index)

    def test_n_edges_attribute_matches(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        assert acc.n_edges == len(acc.edge_index)


# =============================================================================
# 3. ADMISSIBILITY MASK & RULES
# =============================================================================

class TestAdmissibilityRules:

    def test_no_p_to_p_removes_correct_edges(self, root_simple):
        """Mining(P)→Smelting(P) must be absent; Mining(P)→Ore(F) must be present."""
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])

        mining_id   = acc.node_to_id[("Mining",  "P", "A")]
        ore_id      = acc.node_to_id[("Ore",     "F", "A")]
        smelting_id = acc.node_to_id[("Smelting","P", "A")]

        ei = acc.edge_index
        assert not ((ei["from_id"] == mining_id)  & (ei["to_id"] == smelting_id)).any()
        assert     ((ei["from_id"] == mining_id)  & (ei["to_id"] == ore_id)).any()

    def test_no_f_to_f_removes_correct_edges(self, root_simple):
        acc    = AccountingStructure(id="t", root=root_simple, rules=[rule_no_f_to_f])
        ore_id = acc.node_to_id[("Ore", "F", "A")]
        ei     = acc.edge_index
        assert not ((ei["from_id"] == ore_id) & (ei["to_id"] == ore_id)).any()

    def test_no_self_loop_rule(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_self_loop])
        ei  = acc.edge_index
        assert not (ei["from_id"] == ei["to_id"]).any()

    def test_no_cross_region_rule(self, root_multi_region):
        acc = AccountingStructure(id="t", root=root_multi_region, rules=[rule_no_cross_region])
        ei  = acc.edge_index
        node_region = {nid: t[2] for t, nid in acc.node_to_id.items()}
        from_regions = ei["from_id"].map(node_region)
        to_regions   = ei["to_id"].map(node_region)
        assert (from_regions == to_regions).all()

    def test_combined_rules_intersection(self, root_simple):
        acc_both = AccountingStructure(id="t", root=root_simple,
                                       rules=[rule_no_p_to_p, rule_no_self_loop])
        acc_one  = AccountingStructure(id="t", root=root_simple,
                                       rules=[rule_no_p_to_p])
        assert len(acc_both.edge_index) <= len(acc_one.edge_index)

    def test_always_false_rule_yields_zero_edges(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_always_false])
        assert len(acc.edge_index) == 0

    def test_no_rules_yields_full_edge_set(self, root_simple):
        acc      = AccountingStructure(id="t", root=root_simple, rules=[])
        expected = 3 * 3 * 1 * 1
        assert len(acc.edge_index) == expected

    def test_A_nodes_shape(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        assert acc.A_nodes.shape == (3, 3)

    def test_A_tensor_shape(self, root_simple):
        """Full tensor must be 4-D: (n_nodes, n_nodes, n_timeps, n_valtns)."""
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        assert acc.A.shape == (3, 3, 1, 1)

    def test_A_tensor_shape_multi(self, root_multi_time):
        acc = AccountingStructure(id="t", root=root_multi_time)
        assert acc.A.shape == (2, 2, 2, 2)

    def test_A_nodes_consistent_with_edge_index(self, root_simple):
        """Every (from_id, to_id) in edge_index must be 1 in A_nodes."""
        acc     = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        A_dense = acc.A_nodes.todense()
        for _, row in acc.edge_index.iterrows():
            assert A_dense[int(row["from_id"]), int(row["to_id"])] == 1

    def test_A_tensor_consistent_with_edge_index(self, root_multi_time):
        """Every edge in edge_index must be 1 in the full tensor."""
        acc     = AccountingStructure(id="t", root=root_multi_time)
        A_dense = acc.A.todense()
        for _, row in acc.edge_index.iterrows():
            t_idx = acc.time_to_idx[row["time"]]
            v_idx = acc.val_to_idx[row["valuation"]]
            assert A_dense[int(row["from_id"]), int(row["to_id"]), t_idx, v_idx] == 1

    def test_A_tensor_nnz_equals_n_edges(self, root_simple):
        """Number of nonzeros in A must equal number of admissible edges.
        root_simple has Mining(P), Ore(F), Smelting(P) — rule_no_p_to_p blocks
        Mining→Smelting and Smelting→Mining, leaving 7 of 9 node pairs × 1 time
        × 1 valuation = 7 admissible edges, matching A.nnz exactly.
        """
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        assert acc.A.nnz == acc.n_edges

    def test_rule_short_circuits_on_first_failure(self, root_simple):
        call_log = []

        def counting_rule(ni, nj):
            call_log.append(1)
            return False

        def should_not_be_called(ni, nj):
            raise AssertionError("Second rule was called despite first blocking")

        acc = AccountingStructure(id="t", root=root_simple,
                                  rules=[counting_rule, should_not_be_called])
        assert len(acc.edge_index) == 0


# =============================================================================
# 4. TIME AND VALUATION DIMENSIONS
# =============================================================================

class TestTimeValuation:

    def test_all_time_steps_present(self, root_multi_time):
        acc = AccountingStructure(id="t", root=root_multi_time)
        assert set(acc.edge_index["time"].unique()) == {0, 1}

    def test_all_valuations_present(self, root_multi_time):
        acc = AccountingStructure(id="t", root=root_multi_time)
        assert set(acc.edge_index["valuation"].unique()) == {"mass", "value"}

    def test_edge_count_scales_with_time(self):
        """Doubling time steps must double edge count."""
        root_1t = {
            "nodes": pd.DataFrame({"sector": ["A","B"], "entity": ["P","F"], "region": ["X","X"]}),
            "time_range": [0], "valuation": ["mass"]
        }
        root_2t = {
            "nodes": pd.DataFrame({"sector": ["A","B"], "entity": ["P","F"], "region": ["X","X"]}),
            "time_range": [0, 1], "valuation": ["mass"]
        }
        acc_1 = AccountingStructure(id="t", root=root_1t)
        acc_2 = AccountingStructure(id="t", root=root_2t)
        assert len(acc_2.edge_index) == 2 * len(acc_1.edge_index)

    def test_edge_count_scales_with_valuation(self):
        """Doubling valuations must double edge count."""
        root_1v = {
            "nodes": pd.DataFrame({"sector": ["A","B"], "entity": ["P","F"], "region": ["X","X"]}),
            "time_range": [0], "valuation": ["mass"]
        }
        root_2v = {
            "nodes": pd.DataFrame({"sector": ["A","B"], "entity": ["P","F"], "region": ["X","X"]}),
            "time_range": [0], "valuation": ["mass", "value"]
        }
        acc_1 = AccountingStructure(id="t", root=root_1v)
        acc_2 = AccountingStructure(id="t", root=root_2v)
        assert len(acc_2.edge_index) == 2 * len(acc_1.edge_index)


# =============================================================================
# 5. DECODE EDGE
# =============================================================================

class TestDecodeEdge:

    def test_decode_edge_returns_correct_tuple(self, root_simple):
        acc  = AccountingStructure(id="t", root=root_simple)
        row  = acc.edge_index.iloc[0]
        decoded = acc.decode_edge(int(row["edge_id"]))

        assert decoded[0] == acc.decode_node(int(row["from_id"]))
        assert decoded[1] == acc.decode_node(int(row["to_id"]))
        assert decoded[2] == row["time"]
        assert decoded[3] == row["valuation"]

    def test_all_edges_decodable(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple)
        for eid in acc.edge_index["edge_id"]:
            decoded = acc.decode_edge(int(eid))
            assert len(decoded) == 4


# =============================================================================
# 6. COMPRESS
# =============================================================================

class TestCompress:

    def test_compress_dense_4d(self, root_simple):
        acc   = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        shape = (acc.n_nodes, acc.n_nodes, acc.n_timeps, acc.n_valtns)
        F     = np.arange(np.prod(shape), dtype=float).reshape(shape)
        out   = acc.compress(F)
        assert len(out) == len(acc.edge_index)

    def test_compress_sparse_4d(self, root_simple):
        acc   = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        shape = (acc.n_nodes, acc.n_nodes, acc.n_timeps, acc.n_valtns)
        F     = sparse.COO(
            coords=np.zeros((4, 1), dtype=np.int64),
            data=np.array([1.0]),
            shape=shape,
        )
        out = acc.compress(F)
        assert len(out) == len(acc.edge_index)

    def test_compress_values_match_tensor(self, root_simple):
        acc   = AccountingStructure(id="t", root=root_simple)
        shape = (acc.n_nodes, acc.n_nodes, acc.n_timeps, acc.n_valtns)
        F     = np.arange(np.prod(shape), dtype=float).reshape(shape)
        out   = acc.compress(F)
        for k, row in acc.edge_index.iterrows():
            t_idx = acc.time_to_idx[row["time"]]
            v_idx = acc.val_to_idx[row["valuation"]]
            expected = F[int(row["from_id"]), int(row["to_id"]), t_idx, v_idx]
            assert out[k] == expected


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
            "time_range": [0, 1],
            "valuation":  ["mass", "value"],
        }
        return AccountingStructure(
            id="save_test", root=root,
            rules=[rule_no_p_to_p],
            storage_dir=storage_dir
        )

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
        pd.testing.assert_frame_equal(
            loaded.edge_index.reset_index(drop=True),
            acc.edge_index.reset_index(drop=True),
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
        )

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            AccountingStructure.load("nonexistent_id", storage_dir=str(tmp_path))


# =============================================================================
# 8. TIME / VALUATION DEDUPLICATION
# =============================================================================

class TestDeduplication:

    def test_duplicate_time_deduplicated(self, root_with_duplicates):
        acc = AccountingStructure(id="t", root=root_with_duplicates)
        assert acc.time_range == [0, 1], (
            f"Expected [0, 1] after dedup, got {acc.time_range}"
        )

    def test_duplicate_valuation_deduplicated(self, root_with_duplicates):
        acc = AccountingStructure(id="t", root=root_with_duplicates)
        assert acc.valuation == ["mass"], (
            f"Expected ['mass'] after dedup, got {acc.valuation}"
        )

    def test_duplicate_time_does_not_multiply_edges(self, root_with_duplicates):
        """With duplicates removed, edge count must equal the deduplicated count."""
        acc = AccountingStructure(id="t", root=root_with_duplicates)
        # 2 nodes × 2 nodes × 2 unique times × 1 unique valuation = 8
        assert len(acc.edge_index) == 2 * 2 * 2 * 1

    def test_order_preserved_after_dedup(self, root_with_duplicates):
        """First occurrence of each value must be retained in order."""
        acc = AccountingStructure(id="t", root=root_with_duplicates)
        assert acc.time_range[0] == 0
        assert acc.time_range[1] == 1


# =============================================================================
# 9. RULE CLASSIFICATION (node vs edge)
# =============================================================================

class TestRuleClassification:

    def test_node_rule_classified_correctly(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple, rules=[rule_no_p_to_p])
        assert rule_no_p_to_p in acc._node_rules
        assert rule_no_p_to_p not in acc._edge_rules

    def test_edge_rule_classified_correctly(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple, rules=[edge_rule_only_time_0])
        assert edge_rule_only_time_0 in acc._edge_rules
        assert edge_rule_only_time_0 not in acc._node_rules

    def test_mixed_rules_classified_independently(self, root_simple):
        acc = AccountingStructure(
            id="t", root=root_simple,
            rules=[rule_no_p_to_p, edge_rule_only_time_0]
        )
        assert rule_no_p_to_p     in acc._node_rules
        assert edge_rule_only_time_0 in acc._edge_rules
        assert len(acc._node_rules) == 1
        assert len(acc._edge_rules) == 1

    def test_no_rules_both_lists_empty(self, root_simple):
        acc = AccountingStructure(id="t", root=root_simple, rules=[])
        assert acc._node_rules == []
        assert acc._edge_rules == []


# =============================================================================
# 10. EDGE-LEVEL RULES
# =============================================================================

class TestEdgeRules:

    def test_edge_rule_filters_by_time(self, root_multi_time):
        """edge_rule_only_time_0 must halve the edge count vs no rules."""
        acc_all  = AccountingStructure(id="t", root=root_multi_time)
        acc_t0   = AccountingStructure(
            id="t", root=root_multi_time, rules=[edge_rule_only_time_0]
        )
        # Only time=0 survives; 2 valuations remain → half total edges
        assert len(acc_t0.edge_index) == len(acc_all.edge_index) // 2

    def test_edge_rule_filters_by_valuation(self, root_multi_time):
        """edge_rule_only_mass must halve the edge count vs no rules."""
        acc_all  = AccountingStructure(id="t", root=root_multi_time)
        acc_mass = AccountingStructure(
            id="t", root=root_multi_time, rules=[edge_rule_only_mass]
        )
        assert len(acc_mass.edge_index) == len(acc_all.edge_index) // 2

    def test_edge_rule_always_false_yields_zero_edges(self, root_simple):
        acc = AccountingStructure(
            id="t", root=root_simple, rules=[edge_rule_always_false]
        )
        assert len(acc.edge_index) == 0

    def test_node_rule_blocks_before_edge_rule_evaluated(self, root_simple):
        """If a node rule already blocks a pair, the edge rule must not be called."""
        edge_rule_calls = []

        def tracking_edge_rule(ni, nj, time, valuation):
            edge_rule_calls.append((ni, nj, time, valuation))
            return True  # would allow everything if reached

        acc = AccountingStructure(
            id="t", root=root_simple,
            rules=[rule_always_false, tracking_edge_rule]
        )
        # rule_always_false (node rule) blocks all pairs → edge rule never fires
        assert len(acc.edge_index) == 0
        assert edge_rule_calls == [], (
            f"Edge rule was called {len(edge_rule_calls)} times despite node rule blocking all pairs"
        )

    def test_node_mask_unaffected_by_edge_rules(self, root_simple):
        """A_nodes reflects node rules only — edge rules must not alter it."""
        acc_node_only = AccountingStructure(id="t", root=root_simple,
                                            rules=[rule_no_p_to_p])
        acc_with_edge = AccountingStructure(id="t", root=root_simple,
                                            rules=[rule_no_p_to_p, edge_rule_only_time_0])
        # A_nodes must be identical regardless of edge rules
        assert (acc_node_only.A_nodes - acc_with_edge.A_nodes).nnz == 0

    def test_combined_node_and_edge_rules(self, root_multi_time):
        """Node rule removes P→P pairs; edge rule removes time≠0. Both must apply."""
        acc = AccountingStructure(
            id="t", root=root_multi_time,
            rules=[rule_no_p_to_p, edge_rule_only_time_0]
        )
        ei = acc.edge_index

        # No P→P edges
        node_region = {nid: t for t, nid in acc.node_to_id.items()}
        for _, row in ei.iterrows():
            ni = acc.decode_node(int(row["from_id"]))
            nj = acc.decode_node(int(row["to_id"]))
            assert not (ni[1] == "P" and nj[1] == "P"), "P→P edge found"

        # Only time=0
        assert (ei["time"] == 0).all(), "Non-zero time edge found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])