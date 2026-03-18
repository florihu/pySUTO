import numpy as np
import pandas as pd

import pytest

from pysuto.acc.AccountingStructure import AccountingStructure
from pysuto.weight.WeightMatrix import WeightMatrix


# =====================================================
# FIXTURE: ACCOUNTING STRUCTURE
# =====================================================
@pytest.fixture
def setup_acc_struct():

    acc_id = "test_structure_full"

    # Cartesian product (important fix!)
    sectors = ["Mining", "Smelting", "Refining"]
    entities = ["P", "F", "R"]
    regions = ["A", "B"]

    root_nodes = pd.DataFrame(
        [(s, e, r) for s in sectors for e in entities for r in regions],
        columns=["sector", "entity", "region"]
    )

    time_range = [0, 1]
    valuation = ["mass", "copper_c"]

    def no_same_entity(ni, nj):
        return ni["entity"] != nj["entity"]

    acc_struct = AccountingStructure(
        id=acc_id,
        root={
            "nodes": root_nodes,
            "time_range": time_range,
            "valuation": valuation
        },
        rules=[no_same_entity]
    )

    return acc_struct


# =====================================================
# FIXTURE: CONCORDANCE
# =====================================================
@pytest.fixture
def setup_concordance(setup_acc_struct):

    acc_struct = setup_acc_struct

    # =====================================================
    # 1. DEFINE SIMPLE SOURCE SYSTEM (EXTERNAL CLASSIFICATION)
    # =====================================================

    # Source classification (DIFFERENT naming!)
    src_edges = pd.DataFrame([
        # Edge 1
        {
            "src_sector_from": "Mine",
            "src_entity_from": "P",
            "src_region_from": "A",
            "src_sector_to":   "Factory",
            "src_entity_to":   "F",
            "src_region_to":   "A",
            "src_time": 0,
            "src_valuation": "mass"
        },
        # Edge 2
        {
            "src_sector_from": "Mine",
            "src_entity_from": "P",
            "src_region_from": "A",
            "src_sector_to":   "Export",
            "src_entity_to":   "R",
            "src_region_to":   "B",
            "src_time": 0,
            "src_valuation": "mass"
        }
    ])

    # =====================================================
    # 2. DEFINE SOURCE FLOWS
    # =====================================================

    # 2 edges → 2 values
    F_src = np.array([100.0, 50.0])

    # =====================================================
    # 3. DEFINE CONCORDANCE TABLE (SRC → ROOT)
    # =====================================================

    con_tab = []

    # ---- Edge 1 maps to 2 root edges ----
    con_tab += [
        {
            **src_edges.iloc[0].to_dict(),

            "root_sector_from": "Mining",
            "root_entity_from": "P",
            "root_region_from": "A",
            "root_sector_to":   "Smelting",
            "root_entity_to":   "F",
            "root_region_to":   "A",
            "root_time": 0,
            "root_valuation": "mass",

            "weight_value": 0.6,
            "weight_mode": "user"
        },
        {
            **src_edges.iloc[0].to_dict(),

            "root_sector_from": "Mining",
            "root_entity_from": "P",
            "root_region_from": "A",
            "root_sector_to":   "Refining",
            "root_entity_to":   "R",
            "root_region_to":   "A",
            "root_time": 0,
            "root_valuation": "mass",

            "weight_value": 0.4,
            "weight_mode": "user"
        }
    ]

    # ---- Edge 2 maps fully to 1 root edge ----
    con_tab += [
        {
            **src_edges.iloc[1].to_dict(),

            "root_sector_from": "Mining",
            "root_entity_from": "P",
            "root_region_from": "A",
            "root_sector_to":   "Refining",
            "root_entity_to":   "R",
            "root_region_to":   "B",
            "root_time": 0,
            "root_valuation": "mass",

            "weight_value": 1.0,
            "weight_mode": "user"
        }
    ]

    con_tab_df = pd.DataFrame(con_tab)

    # =====================================================
    # 4. BUILD CONCORDANCE
    # =====================================================

    concordance = WeightMatrix(con_tab_df, acc_struct)

    return concordance, F_src


# =====================================================
# TEST
# =====================================================
def test_concordance_apply(setup_acc_struct, setup_concordance, tmp_path):

    acc_struct = setup_acc_struct
    concordance, F_src = setup_concordance

    # Apply
    root_result = concordance.apply(F_src)

    # ===========================
    # ASSERTIONS
    # ===========================

    # 1. Shape: vector over ROOT edges
    assert root_result.shape[0] == len(acc_struct.edge_index)

    # 2. Weight matrix has entries
    assert concordance.weight_matrix.nnz > 0

    # 3. Save / load
    save_path = tmp_path / "test_concordance.pkl"
    concordance.save(path=str(save_path))

    concordance_loaded = WeightMatrix.load(
        id=str(save_path).replace(".pkl", ""),
        acc=acc_struct
    )

    assert concordance_loaded.weight_matrix.shape == concordance.weight_matrix.shape

    # ===========================
    # WEIGHT SUM CHECK (per src_edge)
    # ===========================
    src_cols = [
        "src_sector_from","src_entity_from","src_region_from",
        "src_sector_to","src_entity_to","src_region_to",
        "src_time","src_valuation"
    ]

    weights_sum = concordance.tab.groupby(src_cols)["weight_value"].sum()
    np.testing.assert_allclose(weights_sum.values, 1.0)

    # ===========================
    # 🔥 FLOW CONSERVATION CHECK
    # ===========================
    np.testing.assert_allclose(
        F_src.sum(),
        root_result.sum(),
        rtol=1e-6,
        atol=1e-9
    )

if __name__ == "__main__":
    pytest.main([__file__])