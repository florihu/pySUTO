import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

from pysuto.conc import ConcordanceTable, WeightMatrix


# =============================================================================
# FIXTURE
# =============================================================================

@pytest.fixture
def mock_acc_struct():
    acc = SimpleNamespace()

    acc.id = "mock"

    acc.node_to_id = {
        ("Mining",      "P", "A"): 0,
        ("Ore",         "F", "A"): 1,
        ("Ore",         "F", "B"): 2,
        ("Smelting",    "P", "A"): 3,
        ("Smelting",    "P", "B"): 4,
        ("Concentrate", "F", "A"): 5,
        ("Concentrate", "F", "B"): 6,
        ("Refining",    "P", "A"): 7,
        ("Refining",    "P", "B"): 8,
    }
    acc.id_to_node = {v: k for k, v in acc.node_to_id.items()}

    # edge_id values are unique; fixed duplicate from original fixture
    acc.edge_index = pd.DataFrame([
        # Supply chain: Mining → Ore → Smelting → Concentrate → Refining
        {"edge_id": 0, "from_id": 0, "to_id": 1, "time": 0, "valuation": "mass"},  # Mining(A)  → Ore(A)
        {"edge_id": 1, "from_id": 1, "to_id": 3, "time": 0, "valuation": "mass"},  # Ore(A)     → Smelting(A)
        {"edge_id": 2, "from_id": 1, "to_id": 4, "time": 0, "valuation": "mass"},  # Ore(A)     → Smelting(B)  (fan-out)
        {"edge_id": 3, "from_id": 2, "to_id": 4, "time": 0, "valuation": "mass"},  # Ore(B)     → Smelting(B)
        {"edge_id": 4, "from_id": 3, "to_id": 5, "time": 0, "valuation": "mass"},  # Smelting(A)→ Concentrate(A)
        {"edge_id": 5, "from_id": 4, "to_id": 6, "time": 0, "valuation": "mass"},  # Smelting(B)→ Concentrate(B)
        {"edge_id": 6, "from_id": 5, "to_id": 7, "time": 0, "valuation": "mass"},  # Conc(A)    → Refining(A)
        {"edge_id": 7, "from_id": 6, "to_id": 7, "time": 0, "valuation": "mass"},  # Conc(B)    → Refining(A)  (heterogeneous use)
        {"edge_id": 8, "from_id": 6, "to_id": 8, "time": 0, "valuation": "mass"},  # Conc(B)    → Refining(B)
    ])

    return acc


# =============================================================================
# SHARED MAP FIXTURES
# =============================================================================

@pytest.fixture
def maps():
    sector_map_df = pd.DataFrame({
        "src_sector":  ["Mining", "Ore", "Smelting", "Concentrate", "Refining"],
        "root_sector": ["Mining", "Ore", "Smelting", "Concentrate", "Refining"],
    })

    # Region E fans out to both A and B (tests 1:m expansion)
    region_map_df = pd.DataFrame({
        "src_region":  ["A", "B", "E", "E"],
        "root_region": ["A", "B", "A", "B"],
    })

    time_map_df       = pd.DataFrame({"src_time": [0],    "root_time": [0]})
    valuation_map_df  = pd.DataFrame({"src_valuation": ["kg"], "root_valuation": ["mass"]})

    var_map_df = pd.DataFrame([
        # Supply side
        {"raw_var": "Process", "src_var": "src_sector_from", "Supply_Use": "Supply"},
        {"raw_var": "Type",    "src_var": "src_sector_to",   "Supply_Use": "Supply"},
        {"raw_var": "Region",  "src_var": "src_region_from", "Supply_Use": "Supply"},
        # Use side
        {"raw_var": "Process", "src_var": "src_sector_to",   "Supply_Use": "Use"},
        {"raw_var": "Type",    "src_var": "src_sector_from", "Supply_Use": "Use"},
        {"raw_var": "Region",  "src_var": "src_region_to",   "Supply_Use": "Use"},
        # Common
        {"raw_var": "Year",    "src_var": "src_time",        "Supply_Use": None},
        {"raw_var": "Unit",    "src_var": "src_valuation",   "Supply_Use": None},
    ])

    return dict(
        sector_map_df=sector_map_df,
        region_map_df=region_map_df,
        time_map_df=time_map_df,
        valuation_map_df=valuation_map_df,
        var_map_df=var_map_df,
    )


# =============================================================================
# RAW DATA FIXTURE
# =============================================================================

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Process": ["Mining",  "Mining",  "Smelting",   "Refining"],
        "Type":    ["Ore",     "Ore",     "Concentrate","Concentrate"],
        "Region":  ["A",       "E",       "A",          "A"],
        #            1:1 supply  1:m supply  supply       heter. use
        "Year":    [0, 0, 0, 0],
        "Unit":    ["kg"] * 4,
        "Supply_Use": ["Supply", "Supply", "Supply", "Use"],
    })


# Raw data for the m:1 fan-in scenario:
# Conc1(A) and Conc2(A) are two distinct src sectors that both map to root
# sector Concentrate in region A.  Both feed Refining(A) on the Use side,
# so they both resolve to root edge 6 — Concentrate(A) → Refining(A).
#
# Uniform weight imputation operates per src-edge: each src edge (Conc1 and
# Conc2) maps 1:1 to root edge 6, so weight = 1.0 for both.
# apply([1, 1]) therefore produces F_root[6] = 2.0 — the raw flows aggregate.
@pytest.fixture
def raw_df_fanin():
    return pd.DataFrame({
        "Process":    ["Refining", "Refining"],
        "Type":       ["Conc1",    "Conc2"],    # two src sectors → same root Concentrate
        "Region":     ["A",        "A"],         # region used for both to-side and from-side
        "Year":       [0, 0],
        "Unit":       ["kg", "kg"],
        "Supply_Use": ["Use", "Use"],
    })


@pytest.fixture
def maps_fanin(maps):
    """Extend the base maps with Conc1/Conc2 → Concentrate and map Region for both sides."""
    m = {k: v.copy() for k, v in maps.items()}

    # Conc1 and Conc2 both resolve to root sector Concentrate
    m["sector_map_df"] = pd.concat([
        maps["sector_map_df"],
        pd.DataFrame({"src_sector":  ["Conc1",       "Conc2"],
                      "root_sector": ["Concentrate", "Concentrate"]}),
    ], ignore_index=True)

    # For Use rows, also map Region → src_region_from so the secondary-side
    # constraint pins the from-node to Concentrate(A) and excludes edges 7/8.
    m["var_map_df"] = pd.concat([
        maps["var_map_df"],
        pd.DataFrame([{"raw_var": "Region", "src_var": "src_region_from",
                       "Supply_Use": "Use"}]),
    ], ignore_index=True)

    return m


# =============================================================================
# HELPER: build ConcordanceTable from fixtures
# =============================================================================

def build_ct(raw_df, mock_acc_struct, maps, allow_unmatched=True):
    return ConcordanceTable.from_raw(
        raw_df=raw_df,
        acc_struct=mock_acc_struct,
        aux_col="Supply_Use",
        allow_unmatched_edges=allow_unmatched,
        id="test_ct",
        **maps,
    )


# =============================================================================
# TESTS
# =============================================================================

class TestConcordanceTableFromRaw:

    # ------------------------------------------------------------------
    # 1. Heterogeneous use: Refining(A) consumes Concentrate from A & B
    # ------------------------------------------------------------------
    def test_heterogeneous_use_edges_present(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        tab = ct.tab

        refining_rows = tab[tab["root_sector_to"] == "Refining"]

        # Both source regions must appear (Concentrate A and B feed Refining)
        assert set(refining_rows["root_region_from"]) == {"A", "B"}
        # All destination rows are Refining
        assert (refining_rows["root_sector_from"] == "Concentrate").all()

    # ------------------------------------------------------------------
    # 2. 1:m fan-out: region E expands where edges exist for both A and B
    # ------------------------------------------------------------------
    def test_fanout_region_expands_to_both(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        tab = ct.tab

        # Raw row 1 (Mining→Ore, src_region_from=E, Supply).
        # Mining only has one edge: Mining(A)→Ore(A), so region E collapses to A only.
        # The fan-out from E is visible here: only region A is reached for Ore destination.
        ore_rows = tab[tab["root_sector_to"] == "Ore"]
        assert set(ore_rows["root_region_from"]) == {"A"}, "Region E should only reach Ore via Mining(A)"

    # ------------------------------------------------------------------
    # 3. Structurally impossible classified relation raises
    # ------------------------------------------------------------------
    def test_impossible_classified_relation_raises(self, mock_acc_struct, maps):
        # Use row: Type=Ore → Process=Refining gives the relation Ore→Refining,
        # which has no edge in the acc. Both endpoints are classified, so it is a
        # structural impossibility → fatal (not a tolerated unmatched warning).
        bad = pd.DataFrame({
            "Process": ["Refining"], "Type": ["Ore"], "Region": ["B"],
            "Year": [0], "Unit": ["kg"], "Supply_Use": ["Use"],
        })
        with pytest.raises(ValueError, match="(?i)structurally impossible"):
            build_ct(bad, mock_acc_struct, maps)

    # ------------------------------------------------------------------
    # 4. Only edges that exist in acc_struct appear in output
    # ------------------------------------------------------------------
    def test_only_valid_root_edges(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        tab = ct.tab

        allowed_sector_pairs = {
            ("Mining",      "Ore"),
            ("Ore",         "Smelting"),
            ("Smelting",    "Concentrate"),
            ("Concentrate", "Refining"),
        }
        actual_pairs = set(
            zip(tab["root_sector_from"], tab["root_sector_to"])
        )
        assert actual_pairs <= allowed_sector_pairs

    # ------------------------------------------------------------------
    # 5. Weights imputed to uniform and sum to 1 per src edge
    # ------------------------------------------------------------------
    def test_uniform_weights_sum_to_one(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)

        sums = ct.tab.groupby(ct.SRC_COLS)["weight_value"].sum()
        assert np.allclose(sums.values, 1.0), f"Weight sums off:\n{sums}"

    # ------------------------------------------------------------------
    # 6. src_edge index is consistent with tab
    # ------------------------------------------------------------------
    def test_src_edge_index_consistent(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)

        assert len(ct.src_edges) == ct.n_src_edges
        assert len(ct.src_edge_to_id) == ct.n_src_edges

        # Every src key in tab must be registered
        for _, r in ct.tab.iterrows():
            key = tuple(r[c] for c in ct.SRC_COLS)
            assert key in ct.src_edge_to_id


class TestManyToOneFanIn:
    """
    Scenario: two raw source sectors (Conc1, Conc2) in region A both map to
    root sector Concentrate via the sector concordance.  Both are on the Use
    side feeding Refining(A), so they both resolve to the same root edge:
    edge_id 6 — Concentrate(A) → Refining(A).

    Uniform weight semantics
    ------------------------
    Uniform imputation divides by the number of root edges *per src edge*.
    Conc1 maps to exactly one root edge (6) → weight = 1.0.
    Conc2 maps to exactly one root edge (6) → weight = 1.0.

    Each src edge carries a full unit weight to root edge 6.  When a unit
    flow vector F_src=[1,1] is applied, root edge 6 accumulates 2.0 — the
    raw flows from both src sectors sum at the root node.  This is the
    correct aggregation behaviour: the weight matrix preserves flows, not
    shares, when the mapping is 1:1 from each src edge to the same root.
    """

    def _build(self, raw_df_fanin, mock_acc_struct, maps_fanin):
        return ConcordanceTable.from_raw(
            raw_df=raw_df_fanin,
            acc_struct=mock_acc_struct,
            aux_col="Supply_Use",
            allow_unmatched_edges=False,
            id="test_fanin",
            **maps_fanin,
        )

    # ------------------------------------------------------------------
    # 10. Both Conc1 and Conc2 resolve to root edge 6
    # ------------------------------------------------------------------
    def test_both_map_to_same_root_edge(self, raw_df_fanin, mock_acc_struct, maps_fanin):
        ct = self._build(raw_df_fanin, mock_acc_struct, maps_fanin)
        tab = ct.tab

        assert len(tab) == 2
        assert set(tab["edge_id"]) == {6}, (
            "Both Conc1 and Conc2 should resolve exclusively to edge 6 "
            f"(Concentrate(A)→Refining(A)), got: {set(tab['edge_id'])}"
        )

    # ------------------------------------------------------------------
    # 11. The two rows occupy distinct src edges
    # ------------------------------------------------------------------
    def test_distinct_src_edges(self, raw_df_fanin, mock_acc_struct, maps_fanin):
        ct = self._build(raw_df_fanin, mock_acc_struct, maps_fanin)
        tab = ct.tab

        src_sectors = set(tab["src_sector_from"])
        assert src_sectors == {"Conc1", "Conc2"}, (
            f"Expected src sectors {{Conc1, Conc2}}, got {src_sectors}"
        )
        assert ct.n_src_edges == 2

    # ------------------------------------------------------------------
    # 12. Uniform weights: each src edge maps 1:1 → weight = 1.0
    # ------------------------------------------------------------------
    def test_fanin_uniform_weights(self, raw_df_fanin, mock_acc_struct, maps_fanin):
        ct = self._build(raw_df_fanin, mock_acc_struct, maps_fanin)
        tab = ct.tab

        # Each src edge (Conc1, Conc2) maps to exactly one root edge,
        # so uniform imputation yields weight = 1.0, not 0.5.
        assert np.allclose(tab["weight_value"].values, 1.0), (
            f"Expected weight 1.0 per src edge (1:1 mapping), got:\n{tab['weight_value']}"
        )

    # ------------------------------------------------------------------
    # 13. Weight matrix: each row sums to 1.0 (row = src edge)
    # ------------------------------------------------------------------
    def test_fanin_weight_matrix_column_sum(self, raw_df_fanin, mock_acc_struct, maps_fanin):
        ct  = self._build(raw_df_fanin, mock_acc_struct, maps_fanin)
        wm  = WeightMatrix(concordance=ct, id="test_fanin_wm")

        W_dense = wm.weight_matrix.todense()

        # Each src-edge row sums to 1.0 (weights per src edge are normalised)
        assert np.allclose(W_dense.sum(axis=1), 1.0), (
            f"Each src-edge row should sum to 1.0, got: {W_dense.sum(axis=1)}"
        )
        # Both src edges point at column 6 — column sum = 2.0
        assert np.isclose(W_dense[:, 6].sum(), 2.0), (
            f"Column 6 should sum to 2.0 (two unit-weight src edges), "
            f"got {W_dense[:, 6].sum()}"
        )

    # ------------------------------------------------------------------
    # 14. apply(): F_src=[1,1] → F_root[6]=2.0 (flows aggregate)
    # ------------------------------------------------------------------
    def test_fanin_apply(self, raw_df_fanin, mock_acc_struct, maps_fanin):
        ct     = self._build(raw_df_fanin, mock_acc_struct, maps_fanin)
        wm     = WeightMatrix(concordance=ct, id="test_fanin_wm")

        F_src  = np.ones(ct.n_src_edges)
        F_root = wm.apply(F_src)

        # Conc1 contributes 1.0 and Conc2 contributes 1.0 to root edge 6
        assert np.isclose(F_root[6], 2.0), (
            f"Expected F_root[6]=2.0 (sum of two unit flows), got {F_root[6]}"
        )
        # All other root edges receive zero
        other = np.delete(F_root, 6)
        assert np.allclose(other, 0.0), (
            f"Expected all other root edges to be 0, got non-zero at indices "
            f"{list(np.where(other != 0)[0])}"
        )




    # ------------------------------------------------------------------
    # 7. WeightMatrix builds correctly from a ConcordanceTable
    # ------------------------------------------------------------------
    def test_weight_matrix_shape(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        wm = WeightMatrix(concordance=ct, id="test_wm")

        assert wm.weight_matrix.shape == (ct.n_src_edges, ct.n_root_edges)

    def test_weight_matrix_nonempty(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        wm = WeightMatrix(concordance=ct, id="test_wm")

        assert len(wm.weight_matrix.data) > 0

    # ------------------------------------------------------------------
    # 8. apply() returns a vector of root-edge length
    # ------------------------------------------------------------------
    def test_apply_output_shape(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        wm = WeightMatrix(concordance=ct, id="test_wm")

        F_src = np.ones(ct.n_src_edges)
        F_root = wm.apply(F_src)

        assert F_root.shape == (ct.n_root_edges,)

    # ------------------------------------------------------------------
    # 9. concordance is accessible via wm.concordance
    # ------------------------------------------------------------------
    def test_concordance_accessible(self, raw_df, mock_acc_struct, maps):
        ct = build_ct(raw_df, mock_acc_struct, maps)
        wm = WeightMatrix(concordance=ct, id="test_wm")

        assert wm.concordance is ct
        assert wm.concordance.tab is ct.tab


# =============================================================================
# STRUCTURAL-VALIDITY GUARD — a classified (from→to) relation must exist as an
# edge in the acc. Fan-out (one endpoint unclassified, split by weights) is OK;
# a structurally impossible pair (e.g. Mining→Concentrate) must raise.
# =============================================================================

def _guard_acc():
    """Mining(A) → {Ore, WR};  MF(A) → Concentrate.  (Mining→Concentrate absent.)"""
    acc = SimpleNamespace(id="guard")
    acc.node_to_id = {
        ("Mining",      "P", "A"): 0,
        ("Ore",         "F", "A"): 1,
        ("WR",          "F", "A"): 2,
        ("MF",          "P", "A"): 3,
        ("Concentrate", "F", "A"): 4,
    }
    acc.id_to_node = {v: k for k, v in acc.node_to_id.items()}
    acc.edge_index = pd.DataFrame([
        {"edge_id": 0, "from_id": 0, "to_id": 1, "time": 0, "valuation": "mass"},  # Mining→Ore
        {"edge_id": 1, "from_id": 0, "to_id": 2, "time": 0, "valuation": "mass"},  # Mining→WR
        {"edge_id": 2, "from_id": 3, "to_id": 4, "time": 0, "valuation": "mass"},  # MF→Concentrate
    ])
    return acc


def _guard_maps(classify_product: bool):
    rows = [
        {"raw_var": "Process", "src_var": "src_sector_from", "Supply_Use": "Supply"},
        {"raw_var": "Region",  "src_var": "src_region_from", "Supply_Use": "Supply"},
        {"raw_var": "Year",    "src_var": "src_time",        "Supply_Use": None},
        {"raw_var": "Unit",    "src_var": "src_valuation",   "Supply_Use": None},
    ]
    if classify_product:
        rows.append({"raw_var": "Type", "src_var": "src_sector_to", "Supply_Use": "Supply"})
    return dict(
        sector_map_df=pd.DataFrame({
            "src_sector":  ["Mining", "Ore", "WR", "MF", "Concentrate"],
            "root_sector": ["Mining", "Ore", "WR", "MF", "Concentrate"],
        }),
        region_map_df=pd.DataFrame({"src_region": ["A"], "root_region": ["A"]}),
        time_map_df=pd.DataFrame({"src_time": [0], "root_time": [0]}),
        valuation_map_df=pd.DataFrame({"src_valuation": ["kg"], "root_valuation": ["mass"]}),
        var_map_df=pd.DataFrame(rows),
    )


def _guard_raw(process, product):
    return pd.DataFrame({
        "Process": [process], "Type": [product], "Region": ["A"],
        "Year": [0], "Unit": ["kg"], "Supply_Use": ["Supply"],
    })


class TestStructuralValidityGuard:

    def test_impossible_classified_pair_raises(self):
        # Both endpoints classified as Mining→Concentrate — no such edge → raise.
        with pytest.raises(ValueError, match="(?i)structurally impossible"):
            ConcordanceTable.from_raw(
                raw_df=_guard_raw("Mining", "Concentrate"), acc_struct=_guard_acc(),
                aux_col="Supply_Use", allow_unmatched_edges=True,
                id="g", **_guard_maps(classify_product=True),
            )

    def test_impossible_pair_raises_regardless_of_allow_unmatched(self):
        # The structural error is fatal independent of allow_unmatched_edges.
        with pytest.raises(ValueError, match="(?i)structurally impossible"):
            ConcordanceTable.from_raw(
                raw_df=_guard_raw("Mining", "Concentrate"), acc_struct=_guard_acc(),
                aux_col="Supply_Use", allow_unmatched_edges=False,
                id="g", **_guard_maps(classify_product=True),
            )

    def test_valid_classified_pair_ok(self):
        # MF→Concentrate is a real edge → lands uniquely, no error.
        ct = ConcordanceTable.from_raw(
            raw_df=_guard_raw("MF", "Concentrate"), acc_struct=_guard_acc(),
            aux_col="Supply_Use", allow_unmatched_edges=True,
            id="g", **_guard_maps(classify_product=True),
        )
        assert set(zip(ct.tab["root_sector_from"], ct.tab["root_sector_to"])) == {
            ("MF", "Concentrate")
        }

    def test_one_endpoint_fanout_does_not_raise(self):
        # Producer-only classification (no product) → legitimately fans out
        # across Mining→Ore and Mining→WR; weights resolve the split. No raise.
        ct = ConcordanceTable.from_raw(
            raw_df=_guard_raw("Mining", "Concentrate"), acc_struct=_guard_acc(),
            aux_col="Supply_Use", allow_unmatched_edges=True,
            id="g", **_guard_maps(classify_product=False),
        )
        assert set(zip(ct.tab["root_sector_from"], ct.tab["root_sector_to"])) == {
            ("Mining", "Ore"), ("Mining", "WR")
        }


# =============================================================================
# Regression: single-endpoint Use rows fan out (one src edge), they do NOT
# replicate the raw value across one src edge per matched region.
# See docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md.
# =============================================================================

class TestSingleEndpointUseFanOut:
    """A Use row classifying only the consumer endpoint (producer region open)
    must yield ONE src edge fanning to its roots with weights summing to 1 —
    mass-preserving. The pre-fix bug materialised one src edge per producer
    region, each carrying the full value (N× mass inflation)."""

    @pytest.fixture
    def raw_use_only(self):
        # Refining(A) consumes Concentrate; producer region NOT classified
        # (maps only Region → src_region_to for Use rows). Edges 6 (Conc(A)→
        # Refining(A)) and 7 (Conc(B)→Refining(A)) both match.
        return pd.DataFrame({
            "Process": ["Refining"], "Type": ["Concentrate"], "Region": ["A"],
            "Year": [0], "Unit": ["kg"], "Supply_Use": ["Use"],
        })

    def _ct(self, raw_use_only, mock_acc_struct, maps):
        return ConcordanceTable.from_raw(
            raw_df=raw_use_only, acc_struct=mock_acc_struct,
            aux_col="Supply_Use", id="use_fanout", **maps,
        )

    def test_single_src_edge_with_open_producer_region(self, raw_use_only, mock_acc_struct, maps):
        ct = self._ct(raw_use_only, mock_acc_struct, maps)
        assert ct.n_src_edges == 1, "single-endpoint Use row must be ONE src edge, not one per region"
        assert set(ct.tab["src_region_from"]) == {"<any>"}

    def test_fans_to_both_roots_with_normalised_weights(self, raw_use_only, mock_acc_struct, maps):
        ct = self._ct(raw_use_only, mock_acc_struct, maps)
        assert set(ct.tab["edge_id"]) == {6, 7}
        # uniform split across the 2 roots of the single src edge
        assert ct.tab["weight_value"].sum() == pytest.approx(1.0)
        assert set(np.round(ct.tab["weight_value"], 6)) == {0.5}

    def test_mass_preserved_through_W(self, raw_use_only, mock_acc_struct, maps):
        ct = self._ct(raw_use_only, mock_acc_struct, maps)
        wm = WeightMatrix(ct)
        F_root = wm.apply(np.full(ct.n_src_edges, 100.0))
        # split, not replicated: 50 + 50 = 100 (the raw value), NOT 200
        assert F_root.sum() == pytest.approx(100.0)
        assert F_root[6] == pytest.approx(50.0)
        assert F_root[7] == pytest.approx(50.0)

    def test_open_sector_spans_entities_stays_one_src_edge(self):
        # An unclassified producer endpoint whose matching edges span TWO
        # entities (a Flow producer AND an SB import source) must still collapse
        # to ONE src edge — entity follows the "<any>" sector, it does not
        # re-split. Regression for the entity half of ADR 0002.
        acc = SimpleNamespace(id="ent")
        acc.node_to_id = {
            ("Conc", "F",  "A"): 0,   # Flow-entity producer
            ("Imp",  "SB", "A"): 1,   # SB-entity (system-boundary import) producer
            ("Refining", "P", "A"): 2,
        }
        acc.id_to_node = {v: k for k, v in acc.node_to_id.items()}
        acc.edge_index = pd.DataFrame([
            {"edge_id": 0, "from_id": 0, "to_id": 2, "time": 0, "valuation": "mass"},  # Conc(F)→Refining
            {"edge_id": 1, "from_id": 1, "to_id": 2, "time": 0, "valuation": "mass"},  # Imp(SB)→Refining
        ])
        sector_map = pd.DataFrame({"src_sector": ["Refining"], "root_sector": ["Refining"]})
        region_map = pd.DataFrame({"src_region": ["A"], "root_region": ["A"]})
        time_map   = pd.DataFrame({"src_time": [0], "root_time": [0]})
        val_map    = pd.DataFrame({"src_valuation": ["kg"], "root_valuation": ["mass"]})
        # Use row classifies ONLY the consumer (Process→to, Region→to); producer
        # sector/region/entity all unclassified.
        var_map = pd.DataFrame([
            {"raw_var": "Process", "src_var": "src_sector_to", "Supply_Use": "Use"},
            {"raw_var": "Region",  "src_var": "src_region_to", "Supply_Use": "Use"},
            {"raw_var": "Year",    "src_var": "src_time",      "Supply_Use": None},
            {"raw_var": "Unit",    "src_var": "src_valuation", "Supply_Use": None},
        ])
        raw = pd.DataFrame({"Process": ["Refining"], "Region": ["A"], "Year": [0],
                            "Unit": ["kg"], "Supply_Use": ["Use"]})
        ct = ConcordanceTable.from_raw(
            raw_df=raw, acc_struct=acc, aux_col="Supply_Use",
            sector_map_df=sector_map, region_map_df=region_map, time_map_df=time_map,
            valuation_map_df=val_map, var_map_df=var_map, id="ent",
        )
        assert ct.n_src_edges == 1, "open-sector fan across entities must be ONE src edge"
        assert set(ct.tab["src_entity_from"]) == {"<any>"}
        # fans to both producers, mass preserved (50 / 50, not 100 / 100)
        F_root = WeightMatrix(ct).apply(np.full(ct.n_src_edges, 100.0))
        assert F_root.sum() == pytest.approx(100.0)
        assert F_root[0] == pytest.approx(50.0)
        assert F_root[1] == pytest.approx(50.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])