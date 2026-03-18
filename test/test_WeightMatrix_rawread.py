import pytest
import pandas as pd
import numpy as np

from pysuto.weight.WeightMatrix import WeightMatrix


class MockAccountingStructure:
    """
    Expanded mock to include sectors Mining, Smelting, Extraction and Market
    across regions A, B, C and edges in both directions as needed for tests.
    """
    def __init__(self):
        self.id = "mock_acc"

        sectors = ["Mining", "Smelting", "Extraction", "Market"]
        # map sector -> representative entity code
        ent_map = {"Mining": "P", "Smelting": "F", "Extraction": "P", "Market": "X"}
        regions = ["A", "B", "C"]

        nodes = []
        for s in sectors:
            for r in regions:
                nodes.append((s, ent_map[s], r))

        self.node_to_id = {node: i for i, node in enumerate(nodes)}

        # Build edges:
        # - for each (sector != Market) create sector -> Market (time 0 and 1 / valuations)
        # - also create Market -> sector edges so "use" mappings are covered
        edge_rows = []
        eid = 0
        for s in ["Mining", "Smelting", "Extraction"]:
            for r in regions:
                from_node = (s, ent_map[s], r)
                to_node = ("Market", ent_map["Market"], r)

                # two time slices and two valuations to be safe
                edge_rows.append({
                    "edge_id": eid, "from_id": self.node_to_id[from_node],
                    "to_id": self.node_to_id[to_node], "time": 0, "valuation": "mass"
                })
                eid += 1
                edge_rows.append({
                    "edge_id": eid, "from_id": self.node_to_id[from_node],
                    "to_id": self.node_to_id[to_node], "time": 1, "valuation": "copper_c"
                })
                eid += 1

                # also Market -> sector edges (reverse direction)
                edge_rows.append({
                    "edge_id": eid, "from_id": self.node_to_id[to_node],
                    "to_id": self.node_to_id[from_node], "time": 0, "valuation": "mass"
                })
                eid += 1
                edge_rows.append({
                    "edge_id": eid, "from_id": self.node_to_id[to_node],
                    "to_id": self.node_to_id[from_node], "time": 1, "valuation": "copper_c"
                })
                eid += 1

        self.edge_index = pd.DataFrame(edge_rows)


@pytest.fixture
def mock_acc_struct():
    return MockAccountingStructure()


def test_from_raw_aux_mapping(mock_acc_struct):
    acc = mock_acc_struct

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------
    raw_df = pd.DataFrame({
        "Process": ["Mining", "Extraction", "Smelting", "Extraction"],
        "Type":    ["P", "P", "F", "P"],
        "Region":  ["A", "C", "B", "C"],
        "Year":    [0, 0, 1, 0],
        "Unit":    ["kg", "kg", "ton", "kg"],
        "Supply_Use": ["Supply", "Supply", "Use", "Use"]
    })

    # --------------------------------------------------
    # SECTOR MAP
    # --------------------------------------------------
    sector_map_df = pd.DataFrame({
        "src_sector": ["Mining", "Extraction", "Smelting", "Market", "P", "F"],
        "root_sector": ["Mining", "Mining", "Smelting", "Market", "Mining", "Smelting"]
    })

    # --------------------------------------------------
    # REGION MAP
    # --------------------------------------------------
    region_map_df = pd.DataFrame({
        "src_region": ["A", "B", "C"],
        "root_region": ["A", "B", "A"]
    })

    time_map_df = pd.DataFrame({
        "src_time": [0, 1],
        "root_time": [0, 1]
    })
    valuation_map_df = pd.DataFrame({

        "src_valuation": ["kg", "ton"],
        "root_valuation": ["mass", "copper_c"]
    })


    # --------------------------------------------------
    # VAR MAP (with aux logic)
    # --------------------------------------------------
    var_map_df = pd.DataFrame([
    {"raw_var": "Process", "src_var": "src_sector_from", "Supply_Use": "Supply"},
    {"raw_var": "Type",    "src_var": "src_sector_to", "Supply_Use": "Supply"},
    {"raw_var": "Region",  "src_var": "src_region_from", "Supply_Use": "Supply"},
    
    {"raw_var": "Process", "src_var": "src_sector_to",   "Supply_Use": "Use"},
    {"raw_var": "Type",    "src_var": "src_sector_from",   "Supply_Use": "Use"},
    {"raw_var": "Region",  "src_var": "src_region_to",   "Supply_Use": "Use"},

    {"raw_var": "Year",    "src_var": "src_time",        "Supply_Use": None},
    {"raw_var": "Unit",    "src_var": "src_valuation",   "Supply_Use": None},
    ])


    # --------------------------------------------------
    # BUILD
    # --------------------------------------------------
    wm = WeightMatrix.from_raw(
        raw_df=raw_df,
        acc_struct=acc,

        sector_map_df=sector_map_df,
        region_map_df=region_map_df,
        time_map_df=time_map_df,
        valuation_map_df=valuation_map_df,
        
        var_map_df=var_map_df,
        aux_col="Supply_Use"
    )

    # --------------------------------------------------
    # ASSERT: mapping worked
    # --------------------------------------------------

    # Extraction → Mining
    assert (wm.tab.loc[raw_df["Process"] == "Extraction", "root_sector_from"] == "Mining").all() \
        or (wm.tab.loc[raw_df["Process"] == "Extraction", "root_sector_to"] == "Mining").all()

    # Region C → A
    assert (wm.tab["root_region_from"].isin(["A", "B"])).all()
    assert "A" in wm.tab["root_region_from"].values

    # Valuation mapping
    assert set(wm.tab["root_valuation"]) == {"mass", "copper_c"}

    # Weights
    sums = wm.tab.groupby(wm.src_cols)["weight_value"].sum().values
    assert np.allclose(sums, 1.0)

    # Matrix exists
    assert len(wm.weight_matrix.data) > 0



# todo test not matchin root lables
    

if __name__ == "__main__":
    pytest.main([__file__])