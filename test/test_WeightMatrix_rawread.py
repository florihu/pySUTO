import pytest
import pandas as pd
import numpy as np
import scipy.sparse as sp
from types import SimpleNamespace

from pysuto.weight.WeightMatrix import WeightMatrix



@pytest.fixture
def mock_acc_struct_cardinality():
    acc = SimpleNamespace()

    # Nodes: Mining → Ore → Smelting → Concentrate → Refining
    acc.node_to_id = {
        ("Mining",       "P", "A"): 0,
        ("Ore",          "F", "A"): 1,
        ("Ore",          "F", "B"): 2,
        ("Smelting",     "P", "A"): 3,
        ("Smelting",     "P", "B"): 4,
        ("Concentrate",  "F", "A"): 5,
        ("Concentrate",  "F", "B"): 6,
        ("Refining",     "P", "A"): 7,
        ("Refining",     "P", "B"): 8,
    }
    acc.id_to_node = {v: k for k, v in acc.node_to_id.items()}

    # Edges
    acc.edge_index = pd.DataFrame([
        # Supply side
        {"edge_id": 0, "from_id": 0, "to_id": 1, "time": 0, "valuation": "mass"},  # Mining(A) -> Ore(A)
        {"edge_id": 1, "from_id": 1, "to_id": 3, "time": 0, "valuation": "mass"},  # Ore(A) -> Smelting(A)
        {"edge_id": 2, "from_id": 1, "to_id": 4, "time": 0, "valuation": "mass"},  # Ore(A) -> Smelting(A)
        {"edge_id": 3, "from_id": 2, "to_id": 4, "time": 0, "valuation": "mass"},  # Ore(B) -> Smelting(B)
        {"edge_id": 4, "from_id": 3, "to_id": 5, "time": 0, "valuation": "mass"},  # Smelting(A) -> Concentrate(A)
        {"edge_id": 4, "from_id": 4, "to_id": 6, "time": 0, "valuation": "mass"},  # Smelting(B) -> Concentrate(B)

        # Use side: heterogeneous
        {"edge_id": 6, "from_id": 5, "to_id": 7, "time": 0, "valuation": "mass"},  # Concentrate(A) -> Refining(A)
        {"edge_id": 7, "from_id": 6, "to_id": 7, "time": 0, "valuation": "mass"},  # Concentrate(B) -> Refining(A)
        {"edge_id": 8, "from_id": 6, "to_id": 8, "time": 0, "valuation": "mass"},  # Concentrate(B) -> Refining(B)
    ])

    return acc


def test_from_raw_supply_use_heterogeneous(mock_acc_struct_cardinality):
    acc = mock_acc_struct_cardinality

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------
    raw_df = pd.DataFrame({
        "Process": [
            "Mining",       # Supply: 1:1
            "Mining",       # Supply: 1:m (fan-out via region mapping)
            "Refining",     # Use: m:1 heterogeneous (consumes A & B)
            "Refining",     # Use: m:1 heterogeneous
            "Smelting",     # Edge impossible
        ],
        "Type": [
            "Ore", "Ore", "Concentrate", "Concentrate", "Ore"
        ],
        "Region": [
            "A", "E", "A", "B", "B"  # 'E' → A/B, last row impossible
        ],
        "Year": [0]*5,
        "Unit": ["kg"]*5,
        "Supply_Use": ["Supply", "Supply", "Use", "Use", "Supply"]
    })

    # --------------------------------------------------
    # MAPS
    # --------------------------------------------------
    sector_map_df = pd.DataFrame({
        "src_sector": ["Mining","Ore","Smelting","Concentrate","Refining"],
        "root_sector": ["Mining","Ore","Smelting","Concentrate","Refining"]
    })

    region_map_df = pd.DataFrame({
        "src_region": ["A","B","E", "E"],  # E → A/B for fanout
        "root_region": ["A","B","A", "B"]  # E maps to both A and B (heterogeneous),
    })
    # Add second mapping for heterogeneous
    region_map_df = pd.concat([region_map_df, pd.DataFrame({"src_region":["E"],"root_region":["B"]})], ignore_index=True)

    time_map_df = pd.DataFrame({"src_time":[0], "root_time":[0]})
    valuation_map_df = pd.DataFrame({"src_valuation":["kg"], "root_valuation":["mass"]})

    var_map_df = pd.DataFrame([
        # Supply
        {"raw_var":"Process","src_var":"src_sector_from","Supply_Use":"Supply"},
        {"raw_var":"Type","src_var":"src_sector_to","Supply_Use":"Supply"},
        {"raw_var":"Region","src_var":"src_region_from","Supply_Use":"Supply"},
        # Use
        {"raw_var":"Process","src_var":"src_sector_to","Supply_Use":"Use"},
        {"raw_var":"Type","src_var":"src_sector_from","Supply_Use":"Use"},
        {"raw_var":"Region","src_var":"src_region_to","Supply_Use":"Use"},
        # Common
        {"raw_var":"Year","src_var":"src_time","Supply_Use":None},
        {"raw_var":"Unit","src_var":"src_valuation","Supply_Use":None},
    ])

    wm = WeightMatrix.from_raw(
        raw_df=raw_df,
        acc_struct=acc,
        sector_map_df=sector_map_df,
        region_map_df=region_map_df,
        time_map_df=time_map_df,
        valuation_map_df=valuation_map_df,
        var_map_df=var_map_df,
        aux_col="Supply_Use",
        allow_unmatched_edges=True
    )

    tab = wm.tab.copy()

    # --------------------------------------------------
    # 1) Check that heterogeneous use edges are represented
    # Refining(A) should consume both Concentrate(A) and Concentrate(B)
    # --------------------------------------------------
    ref_a_rows = tab[tab["root_sector_to"]=="Refining"]
    assert set(ref_a_rows["root_sector_from"]) == {"Concentrate"}
    assert set(ref_a_rows["root_region_from"]) == {"A","B"}  # heterogeneous consumption
    assert (ref_a_rows["root_sector_to"]=="Refining").all()
    assert (ref_a_rows["root_region_to"]=="A").all()

    # --------------------------------------------------
    # 2) Check that unmatched edge is captured
    # --------------------------------------------------
    assert 4 in getattr(wm,"unmatched_raw_rows", [])

    # --------------------------------------------------
    # 3) Only allowed root edges appear
    # --------------------------------------------------
    allowed = {
        ("Mining","Ore"), ("Ore","Smelting"), ("Smelting","Concentrate"), ("Concentrate","Refining")
    }
    for _, r in tab.iterrows():
        assert (r["root_sector_from"], r["root_sector_to"]) in allowed
    # --------------------------------------------------
    # 4. m:1 collapse (two use rows → same edge)
    # --------------------------------------------------
    counts = (
        tab.groupby([
            "root_sector_from", "root_region_from",
            "root_sector_to", "root_region_to"
        ])
        .size()
    )

    # Smelting(B) → Refining(A) appears twice (two raw rows collapsed)
    assert counts.loc[("Smelting", "B", "Refining", "A")] == 2

    # --------------------------------------------------
    # 5. No illegal edges
    # --------------------------------------------------
    allowed = {
        ("Mining", "Smelting"),
        ("Smelting", "Refining"),
    }

    for _, r in tab.iterrows():
        assert (r["root_sector_from"], r["root_sector_to"]) in allowed

    # --------------------------------------------------
    # 6. Matrix exists
    # --------------------------------------------------
    assert len(wm.weight_matrix.data) > 0



# todo test not matchin root lables
    

if __name__ == "__main__":
    pytest.main([__file__])