"""
test_constraints.py
===================
Constraint engine test suite.

Fixture topology (toy_acc)
--------------------------
5 nodes:
  0  Mining / Process / A
  1  Ore    / Flow    / A
  2  Mining / Process / B
  3  Ore    / Flow    / B
  4  Waste  / SB      / A      ← system boundary (open endpoint)

Edges (schematic, Mass valuation, time 2020 and 2021):
  supply: Mining/A → Ore/A       (Process→Flow, same region)
  supply: Mining/B → Ore/B       (Process→Flow, same region)
  use:    Ore/A   → Mining/A     (Flow→Process, same region)
  use:    Ore/B   → Mining/B     (Flow→Process, same region)
  waste:  Mining/A → Waste/SB/A  (Process→SB, same region)

Balanced nodes (have ≥1 supply AND ≥1 use edge): Mining/A, Mining/B, Ore/A, Ore/B
SB node (open endpoint, NOT balanced):           Waste/SB/A

Bounds: default l=0, u=1e9 for all edges except one edge with upper=100.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Helpers to build a minimal toy AccountingStructure without file I/O
# ---------------------------------------------------------------------------

def _make_toy_acc(with_bounds: bool = True):
    """
    Return a minimal AccountingStructure-like object with all attributes
    expected by ConstraintEngine and its builders.

    This avoids needing ODS files in tests.
    """
    from pysuto.acc import AccountingStructure, EdgeBounds

    # Build a minimal structure without calling __init__
    acc = AccountingStructure.__new__(AccountingStructure)
    import logging
    acc.logger = logging.getLogger("test_acc")
    acc.id     = "toy"

    # node_index: 5 nodes
    acc.node_index = pd.DataFrame(
        [
            ("Mining", "Process", "A"),
            ("Ore",    "Flow",    "A"),
            ("Mining", "Process", "B"),
            ("Ore",    "Flow",    "B"),
            ("Waste",  "SB",      "A"),
        ],
        columns=["sector", "entity", "region"],
    )
    acc.n_nodes    = 5
    acc.time_range = ["2020", "2021"]
    acc.valuation  = ["Mass"]

    # edge_index (10 edges: 5 edge types × 2 time periods)
    #   supply: Mining/A(0) → Ore/A(1)       [eid 0, 5]
    #   supply: Mining/B(2) → Ore/B(3)       [eid 1, 6]
    #   use:    Ore/A(1)   → Mining/A(0)     [eid 2, 7]
    #   use:    Ore/B(3)   → Mining/B(2)     [eid 3, 8]
    #   waste:  Mining/A(0)→ Waste/SB(4)     [eid 4, 9]
    rows = []
    eid  = 0
    for t in ["2020", "2021"]:
        for (fi, ti) in [(0,1),(2,3),(1,0),(3,2),(0,4)]:
            rows.append({"edge_id": eid, "from_id": fi, "to_id": ti,
                         "time": t, "valuation": "Mass"})
            eid += 1

    acc.edge_index = pd.DataFrame(rows)
    acc.n_edges    = len(acc.edge_index)

    # ruleset-derived masks (computed manually for this topology)
    # supply edges: eids 0,1,5,6  (Process→Flow)
    # use edges:    eids 2,3,7,8  (Flow→Process)
    # waste (→SB):  eids 4,9
    supply_ids  = {0, 1, 5, 6}
    use_ids     = {2, 3, 7, 8}
    sb_ids      = {4, 9}
    all_eids    = np.arange(acc.n_edges)

    acc.masks = {
        "supply":   np.isin(all_eids, list(supply_ids)),
        "use":      np.isin(all_eids, list(use_ids)),
        "tradable": np.zeros(acc.n_edges, dtype=bool),
        "sb":       np.isin(all_eids, list(sb_ids)),
    }

    # RuleSet placeholder (not needed for mask computation above)
    acc.ruleset = None

    # Bounds: all default except edge_id=0 has upper=100
    bounds          = EdgeBounds.__new__(EdgeBounds)
    bounds.n_edges  = acc.n_edges
    bounds.DEFAULT_LOWER = EdgeBounds.DEFAULT_LOWER
    bounds.DEFAULT_UPPER = EdgeBounds.DEFAULT_UPPER
    bounds.l        = np.zeros(acc.n_edges)
    bounds.u        = np.full(acc.n_edges, EdgeBounds.DEFAULT_UPPER)
    if with_bounds:
        bounds.u[0] = 100.0   # one non-default bound
    non_default = (bounds.l != bounds.DEFAULT_LOWER) | (bounds.u != bounds.DEFAULT_UPPER)
    bounds.n_exceptions = int(non_default.sum())
    acc.bounds = bounds

    return acc


def _make_toy_agg_result(acc, seed: int = 42):
    """Return a minimal AggregationResult-like namespace for toy_acc."""
    rng     = np.random.default_rng(seed)
    f_agg   = rng.uniform(1.0, 100.0, size=acc.n_edges)
    sigma_a = rng.uniform(0.1, 10.0, size=acc.n_edges)

    class _R:
        pass

    r           = _R()
    r.f_agg     = f_agg
    r.sigma_agg = sigma_a
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_acc():
    return _make_toy_acc()


@pytest.fixture
def toy_agg_result(toy_acc):
    return _make_toy_agg_result(toy_acc)


# ---------------------------------------------------------------------------
# MassBalanceBuilder
# ---------------------------------------------------------------------------

class TestMassBalanceBuilder:

    def test_mass_balance_rows(self, toy_acc):
        """G rows for a balanced toy solution sum to zero."""
        from pysuto.constraints import MassBalanceBuilder
        builder = MassBalanceBuilder()
        block   = builder.build_static(toy_acc)

        # A feasible balanced solution: supply = use for each node at each time
        # For node Ore/A (node 1): supply edge 0 (→1), use edge 2 (1→)
        # Set f = [10, 10, 10, 10, 0, 10, 10, 10, 10, 0] (supply=use, waste=0)
        f = np.array([10., 10., 10., 10., 0., 10., 10., 10., 10., 0.])
        residuals = block.G_rows @ f - block.c
        assert np.allclose(residuals, 0.0, atol=1e-10), \
            f"Balanced solution should have zero residuals; got {residuals}"

    def test_mass_balance_excludes_sb(self, toy_acc):
        """SB nodes (Waste/SB/A = node 4) produce no G rows."""
        from pysuto.constraints import MassBalanceBuilder
        block    = MassBalanceBuilder().build_static(toy_acc)
        row_meta = block.row_meta
        # No constraint_id should reference node 4
        assert not any("node4" in str(cid) for cid in row_meta["constraint_id"]), \
            "SB node 4 should not appear in any mass balance row"

    def test_mass_balance_cone_type(self, toy_acc):
        """All mass balance rows have cone_type='zero' and is_hard=True."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        assert (block.row_meta["cone_type"] == "zero").all()
        assert block.row_meta["is_hard"].all()

    def test_mass_balance_n_rows(self, toy_acc):
        """Expect one G row per balanced node per timestep (4 nodes × 2 times = 8)."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        # Balanced: Mining/A(0), Ore/A(1), Mining/B(2), Ore/B(3) × 2 timesteps
        assert block.G_rows.shape[0] == 8, \
            f"Expected 8 rows; got {block.G_rows.shape[0]}"

    def test_mass_balance_sigma_c_zero(self, toy_acc):
        """sigma_c is zero for hard constraints."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        assert np.all(block.sigma_c == 0.0)

    def test_mass_balance_row_meta_has_time(self, toy_acc):
        """row_meta carries a `time` column; constraint_id follows the grammar
        `massbal_massbal_{sector}_{region}_{valuation}_{year}`."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        rm = block.row_meta
        assert "time" in rm.columns
        for cid, t, v in zip(rm["constraint_id"], rm["time"], rm["valuation"]):
            assert str(cid).startswith("massbal_massbal_"), cid
            assert str(cid).endswith(f"_{v}_{t}"), \
                f"constraint_id={cid!r} does not end with _{v}_{t}"


# ---------------------------------------------------------------------------
# SumConstraintBuilder
# ---------------------------------------------------------------------------

class TestSumConstraintBuilder:

    def test_sum_from_acc_struct_useful_supply(self, toy_acc, toy_agg_result):
        """Auto-derived useful-supply rows exclude waste (SB) edges."""
        from pysuto.constraints import SumConstraintBuilder
        builder = SumConstraintBuilder.from_acc_struct(toy_acc, sigma_fraction=0.05)
        block   = builder.build_static(toy_acc)
        G       = block.G_rows.toarray()

        # Waste edges (ids 4 and 9) should not appear in any G row
        assert np.all(G[:, 4] == 0.0), "Waste edge 4 should not appear in sum constraints"
        assert np.all(G[:, 9] == 0.0), "Waste edge 9 should not appear in sum constraints"

    def test_from_mask_anchor_groups_sigma_and_skip(self, toy_acc):
        """from_mask_anchor: one row per (region_to, time, valuation) group of
        masked edges; c = Σ anchor over the group; σ_c = sigma_cv·|c|; groups
        whose anchor sums to ~0 are skipped; wrong-year rows deactivate in
        build_dynamic."""
        from pysuto.constraints import SumConstraintBuilder, SIGMA_C_INACTIVE

        n    = toy_acc.n_edges
        # Mask the four supply edges Mining→Ore: 0,1 (t2020) and 5,6 (t2021).
        mask = toy_acc.masks["supply"].copy()

        # Per-edge reference: group (Ore/B, 2021) = eid 6 → 0 ⇒ skipped.
        anchor      = np.zeros(n)
        anchor[0]   = 10.0   # → Ore/A, 2020
        anchor[1]   = 20.0   # → Ore/B, 2020
        anchor[5]   = 30.0   # → Ore/A, 2021
        anchor[6]   = 0.0    # → Ore/B, 2021  (skipped)

        cv      = 0.01
        builder = SumConstraintBuilder.from_mask_anchor(
            toy_acc, mask=mask, anchor_values=anchor,
            group_keys=("region_to", "time", "valuation"),
            sigma_cv=cv, id_prefix="anchor",
        )
        block = builder.build_static(toy_acc)
        rm    = block.row_meta

        # 3 non-zero groups; the zero-anchor group is dropped.
        assert block.G_rows.shape[0] == 3
        assert (rm["type"] == "sum").all()
        assert (rm["variant"] == "anchor").all()
        assert (rm["cone_type"] == "free").all()
        assert (~rm["is_hard"].astype(bool)).all()

        # Each row sums exactly its one masked edge (+1); c and σ match anchor.
        G = block.G_rows.toarray()
        for i, c in enumerate(block.c):
            eids = np.flatnonzero(G[i] != 0.0)
            assert len(eids) == 1
            assert np.isclose(G[i, eids[0]], 1.0)
            assert np.isclose(c, anchor[eids[0]])
            assert np.isclose(block.sigma_c[i], cv * abs(c))

        # build_dynamic at 2020 deactivates the 2021 row; 2020 rows stay tight.
        f   = np.zeros(n)
        s   = np.zeros(n)
        b20 = builder.build_dynamic(toy_acc, f, s, "2020")
        is2021 = (rm["time"].astype(str) == "2021").to_numpy()
        assert np.allclose(b20.sigma_c[is2021], SIGMA_C_INACTIVE)
        assert np.allclose(b20.sigma_c[~is2021], block.sigma_c[~is2021])

    def test_sum_row_meta_has_time(self, toy_acc):
        """SumConstraintBuilder row_meta has a `time` column; constraint_id
        follows the grammar `sum_usefulsupply_{sector}_{region}_{val}_{year}`."""
        from pysuto.constraints import SumConstraintBuilder
        block = SumConstraintBuilder.from_acc_struct(
            toy_acc, sigma_fraction=0.05,
        ).build_static(toy_acc)
        rm = block.row_meta
        assert "time" in rm.columns
        assert rm["time"].notna().all()
        for cid, t, v in zip(rm["constraint_id"], rm["time"], rm["valuation"]):
            assert str(cid).startswith("sum_usefulsupply_"), cid
            assert str(cid).endswith(f"_{v}_{t}")

    def test_sum_sigma_c_varies_per_timestep(self, toy_acc):
        """build_dynamic returns genuinely different σ_c per timestep:
        rows whose spec time matches the call's t are computed normally
        (`fraction_of_flow`); rows from other years are deactivated
        (σ_c = SIGMA_C_INACTIVE)."""
        from pysuto.constraints import SumConstraintBuilder, SIGMA_C_INACTIVE

        f     = np.full(toy_acc.n_edges, 5.0)
        sigma = np.full(toy_acc.n_edges, 1.0)

        builder = SumConstraintBuilder.from_acc_struct(toy_acc, sigma_fraction=0.05)
        times = list(toy_acc.time_range)
        assert len(times) >= 2, "toy_acc needs ≥2 timesteps for this test"

        b_t0 = builder.build_dynamic(toy_acc, f, sigma, times[0])
        b_t1 = builder.build_dynamic(toy_acc, f, sigma, times[1])

        # σ_c should differ across t (different rows active).
        assert not np.allclose(b_t0.sigma_c, b_t1.sigma_c), \
            "σ_c(t0) and σ_c(t1) should not be identical with per-year deactivation"

        # Per-row: spec_time == t0 → active at t0, deactivated at t1.
        rm = b_t0.row_meta
        active_t0 = (rm["time"].astype(str) == str(times[0])).to_numpy()
        active_t1 = (rm["time"].astype(str) == str(times[1])).to_numpy()
        if active_t0.any():
            assert np.allclose(b_t1.sigma_c[active_t0], SIGMA_C_INACTIVE)
        if active_t1.any():
            assert np.allclose(b_t0.sigma_c[active_t1], SIGMA_C_INACTIVE)

    def test_sum_sigma_c_scales_with_magnitude(self, toy_acc):
        """sigma_c from fraction_of_flow doubles when f_agg doubles, for
        rows whose spec time matches the requested timestep. Rows whose
        spec belongs to a different year are deactivated (σ_c =
        SIGMA_C_INACTIVE) and therefore have ratio 1.0."""
        from pysuto.constraints import SumConstraintBuilder, SIGMA_C_INACTIVE

        class _R:
            pass

        r1         = _R()
        r1.f_agg   = np.ones(toy_acc.n_edges) * 10.0
        r1.sigma_agg = np.ones(toy_acc.n_edges)

        r2         = _R()
        r2.f_agg   = np.ones(toy_acc.n_edges) * 20.0
        r2.sigma_agg = np.ones(toy_acc.n_edges)

        builder = SumConstraintBuilder.from_acc_struct(toy_acc, sigma_fraction=0.1)
        b1 = builder.build_dynamic(toy_acc, r1.f_agg, r1.sigma_agg, "2020")
        b2 = builder.build_dynamic(toy_acc, r2.f_agg, r2.sigma_agg, "2020")

        assert b1.G_rows.shape[0] > 0, "Should have at least one dynamic row"

        active = b1.sigma_c < SIGMA_C_INACTIVE * 0.5
        assert active.any(), "Expected at least one row active at t=2020"

        ratio_active = b2.sigma_c[active] / b1.sigma_c[active]
        assert np.allclose(ratio_active, 2.0, rtol=0.01), \
            "σ_c on active rows should double when f_agg doubles"

        # Deactivated rows (spec time != 2020) stay pinned at the sentinel.
        if (~active).any():
            assert np.allclose(b1.sigma_c[~active], SIGMA_C_INACTIVE)
            assert np.allclose(b2.sigma_c[~active], SIGMA_C_INACTIVE)

    def test_sum_from_csv_three_terms(self, toy_acc, toy_agg_result, tmp_path):
        """CSV with 3 terms produces correct G row."""
        import csv
        from pysuto.constraints import SumConstraintBuilder

        csv_path = tmp_path / "sum.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "constraint_id", "description",
                "sector_from", "entity_from", "region_from",
                "sector_to",   "entity_to",   "region_to",
                "time", "valuation",
                "coefficient", "c_k", "sigma_ck_mode", "sigma_ck_value", "sigma_ck_fraction",
            ])
            w.writeheader()
            # term 1: supply Mining/A → Ore/A  at 2020 (edge 0)
            w.writerow({
                "constraint_id": "testsum", "description": "test",
                "sector_from": "Mining", "entity_from": "Process", "region_from": "A",
                "sector_to":   "Ore",    "entity_to":   "Flow",    "region_to":   "A",
                "time": "2020", "valuation": "Mass",
                "coefficient": "1.0", "c_k": "0", "sigma_ck_mode": "sd",
                "sigma_ck_value": "5.0", "sigma_ck_fraction": "",
            })
            # term 2: use Ore/A → Mining/A  at 2020 (edge 2), subtract
            w.writerow({
                "constraint_id": "testsum", "description": "",
                "sector_from": "Ore",    "entity_from": "Flow",    "region_from": "A",
                "sector_to":   "Mining", "entity_to":   "Process", "region_to":   "A",
                "time": "2020", "valuation": "Mass",
                "coefficient": "-1.0", "c_k": "0", "sigma_ck_mode": "sd",
                "sigma_ck_value": "5.0", "sigma_ck_fraction": "",
            })

        builder = SumConstraintBuilder.from_csv(str(csv_path), toy_acc)
        block   = builder.build_static(toy_acc)

        assert block.G_rows.shape[0] >= 1, "Should produce at least 1 row"
        G = block.G_rows.toarray()
        assert block.sigma_c[0] == pytest.approx(5.0)

    def test_sum_cone_type(self, toy_acc):
        """All sum constraint rows have cone_type='free' and is_hard=False."""
        from pysuto.constraints import SumConstraintBuilder
        builder = SumConstraintBuilder.from_acc_struct(toy_acc)
        block   = builder.build_static(toy_acc)
        if block.G_rows.shape[0] > 0:
            assert (block.row_meta["cone_type"] == "free").all()
            assert (~block.row_meta["is_hard"]).all()


# ---------------------------------------------------------------------------
# Per-valuation split (hard rule: no Mass+CC mixing in sum_eq / mass_balance)
# ---------------------------------------------------------------------------

def _make_two_valuation_acc():
    """toy_acc clone with both Mass and CC valuations on the same topology."""
    from pysuto.acc import AccountingStructure, EdgeBounds
    acc = AccountingStructure.__new__(AccountingStructure)
    import logging
    acc.logger = logging.getLogger("test_acc_2val")
    acc.id     = "toy_2val"
    acc.node_index = pd.DataFrame(
        [
            ("Mining", "Process", "A", None),
            ("Ore",    "Flow",    "A", None),
            ("Mining", "Process", "B", None),
            ("Ore",    "Flow",    "B", None),
            ("Waste",  "SB",      "A", "Nature"),
        ],
        columns=["sector", "entity", "region", "entity_sub"],
    )
    acc.n_nodes    = 5
    acc.time_range = ["2020", "2021"]
    acc.valuation  = ["Mass", "CC"]

    rows = []
    eid  = 0
    for t in ["2020", "2021"]:
        for val in ["Mass", "CC"]:
            for (fi, ti) in [(0,1),(2,3),(1,0),(3,2),(0,4)]:
                rows.append({"edge_id": eid, "from_id": fi, "to_id": ti,
                             "time": t, "valuation": val})
                eid += 1
    acc.edge_index = pd.DataFrame(rows)
    acc.n_edges    = len(acc.edge_index)

    # Per-(time, val) bucket of 5 edges: supply (0,1), use (2,3), waste (4).
    ei = acc.edge_index
    from_sector = acc.node_index.loc[ei["from_id"].values, "sector"].values
    to_sector   = acc.node_index.loc[ei["to_id"].values,   "sector"].values
    from_entity = acc.node_index.loc[ei["from_id"].values, "entity"].values
    to_entity   = acc.node_index.loc[ei["to_id"].values,   "entity"].values
    supply = (from_entity == "Process") & np.isin(to_entity, ["Flow", "SB"]) & (to_sector != "Waste")
    use    = np.isin(from_entity, ["Flow", "SB"]) & (to_entity == "Process")
    sb     = (to_entity == "SB")

    acc.masks = {
        "supply":   supply,
        "use":      use,
        "tradable": np.zeros(acc.n_edges, dtype=bool),
        "sb":       sb,
    }
    acc.ruleset = None

    bounds          = EdgeBounds.__new__(EdgeBounds)
    bounds.n_edges  = acc.n_edges
    bounds.DEFAULT_LOWER = EdgeBounds.DEFAULT_LOWER
    bounds.DEFAULT_UPPER = EdgeBounds.DEFAULT_UPPER
    bounds.l        = np.zeros(acc.n_edges)
    bounds.u        = np.full(acc.n_edges, EdgeBounds.DEFAULT_UPPER)
    bounds.n_exceptions = 0
    acc.bounds = bounds
    return acc


@pytest.fixture
def toy_acc_2val():
    return _make_two_valuation_acc()


class TestPerValuationSplit:

    def test_mass_balance_doubles_with_two_valuations(self, toy_acc_2val):
        """Same topology in 2 valuations → 2× the mass-balance rows."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc_2val)
        # Balanced nodes: Mining/A, Ore/A, Mining/B, Ore/B (4)
        # times: 2 ; valuations: 2 → 4 × 2 × 2 = 16
        assert block.G_rows.shape[0] == 16
        assert set(block.row_meta["valuation"]) == {"Mass", "CC"}

    def test_mass_balance_rows_are_unit_coherent(self, toy_acc_2val):
        """No mass_balance G row touches edges of more than one valuation."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc_2val)
        G = block.G_rows.tocsr()
        edge_val = toy_acc_2val.edge_index["valuation"].values
        for i in range(G.shape[0]):
            cols = G.indices[G.indptr[i]: G.indptr[i+1]]
            assert len(set(edge_val[cols])) == 1, \
                f"mass_balance row {i} touches multiple valuations"

    def test_sum_from_acc_struct_splits_per_valuation(self, toy_acc_2val):
        from pysuto.constraints import SumConstraintBuilder
        block = SumConstraintBuilder.from_acc_struct(
            toy_acc_2val, sigma_fraction=0.05,
        ).build_static(toy_acc_2val)
        # from_acc_struct builds rows per balanced-by-supply/use node. The
        # toy fixture has Ore/A and Ore/B balanced (supply lands at Ore;
        # use leaves Ore). Mining nodes have no incoming supply → not
        # balanced by from_acc_struct semantics. So 2 × 2 (times) × 2 (vals) = 8.
        assert block.G_rows.shape[0] == 8
        assert set(block.row_meta["valuation"]) == {"Mass", "CC"}

    def test_sum_subentity_splits_per_valuation(self, toy_acc_2val):
        from pysuto.constraints import SumConstraintBuilder
        block = SumConstraintBuilder.from_acc_struct_subentity(
            toy_acc_2val,
            entity_filter={"Process"},
            rhs_out_entity={"Flow"},
            lhs_neg_out_entity_sub={"Nature"},
        ).build_static(toy_acc_2val)
        # Process nodes: Mining/A, Mining/B at 2 times × 2 valuations = 8.
        assert block.G_rows.shape[0] == 8
        assert set(block.row_meta["valuation"]) == {"Mass", "CC"}

    def test_sum_rows_are_unit_coherent(self, toy_acc_2val):
        from pysuto.constraints import SumConstraintBuilder
        block = SumConstraintBuilder.from_acc_struct(
            toy_acc_2val, sigma_fraction=0.05,
        ).build_static(toy_acc_2val)
        G = block.G_rows.tocsr()
        edge_val = toy_acc_2val.edge_index["valuation"].values
        for i in range(G.shape[0]):
            cols = G.indices[G.indptr[i]: G.indptr[i+1]]
            assert len(set(edge_val[cols])) == 1

    def test_row_meta_carries_node_attrs(self, toy_acc_2val):
        """row_meta picks up sector / region / entity / entity_sub from the
        node the row centers on (decoded from acc.node_index)."""
        from pysuto.constraints import MassBalanceBuilder, SumConstraintBuilder
        mb = MassBalanceBuilder().build_static(toy_acc_2val)
        for col in ("valuation", "sector", "region", "entity", "entity_sub"):
            assert col in mb.row_meta.columns

        # Ore-A row: sector=Ore, region=A, entity=Flow
        ore_a_row = mb.row_meta[
            (mb.row_meta["sector"] == "Ore")
            & (mb.row_meta["region"] == "A")
            & (mb.row_meta["valuation"] == "Mass")
            & (mb.row_meta["time"] == "2020")
        ]
        assert len(ore_a_row) == 1
        assert ore_a_row.iloc[0]["entity"] == "Flow"

        sb = SumConstraintBuilder.from_acc_struct(
            toy_acc_2val, sigma_fraction=0.05,
        ).build_static(toy_acc_2val)
        for col in ("valuation", "sector", "region", "entity"):
            assert col in sb.row_meta.columns


class TestUnitMixingCheck:

    def test_check_raises_on_mixed_sum_row(self, toy_acc_2val, toy_agg_result):
        """Manually inject a sum_eq row that mixes Mass and CC edges; engine
        _check must raise ConstraintUnitMixingError."""
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder,
            ConstraintUnitMixingError,
        )

        # Build the engine + a minimal agg_result for toy_acc_2val.
        agg = _make_toy_agg_result(toy_acc_2val, seed=42)
        engine = ConstraintEngine(
            acc_struct=toy_acc_2val,
            storage_dir=str("/tmp/test_const_runs"),
            sigma_a_cv_min=0.0,
            sigma_a_abs_floor=0.0,
            sigma_c_cv_min=0.0,
            sigma_c_abs_floor=0.0,
        )
        engine.register(MassBalanceBuilder())

        # Run once to obtain a valid baseline run, then mutate G + row_meta
        # to inject a mixed row and re-invoke _check.
        cr = engine.run(
            agg_result=agg,
            timesteps=[str(t) for t in toy_acc_2val.time_range],
            persist=False,
        )
        # Inject a fake sum_eq row that mixes Mass (eid 0) and CC (eid 5).
        n_edges = toy_acc_2val.n_edges
        bad_row = sp.csr_matrix(
            ([1.0, 1.0], ([0, 0], [0, 5])), shape=(1, n_edges),
        )
        cr.G = sp.vstack([cr.G, bad_row], format="csr")
        cr.row_meta = pd.concat([
            cr.row_meta,
            pd.DataFrame([{
                "constraint_id": "fake_mixed",
                "type":          "sum_eq",
                "is_hard":       False,
                "cone_type":     "free",
                "time":          "2020",
                "valuation":     None,
            }]),
        ], ignore_index=True)
        for td in cr.timestep_data.values():
            td["c"]       = np.append(td["c"], 0.0)
            td["sigma_c"] = np.append(td["sigma_c"], 1.0)

        with pytest.raises(ConstraintUnitMixingError, match="mixes valuations"):
            engine._check(cr)

    def test_check_passes_for_ratio_eq_with_mixed_valuations(self, toy_acc_2val, toy_agg_result):
        """Whitelisted ratio_eq rows may span valuations without raising."""
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder,
            ConstraintUnitMixingError,
        )

        agg = _make_toy_agg_result(toy_acc_2val, seed=7)
        engine = ConstraintEngine(
            acc_struct=toy_acc_2val,
            storage_dir=str("/tmp/test_const_runs"),
            sigma_a_cv_min=0.0,
            sigma_a_abs_floor=0.0,
            sigma_c_cv_min=0.0,
            sigma_c_abs_floor=0.0,
        )
        engine.register(MassBalanceBuilder())
        cr = engine.run(
            agg_result=agg,
            timesteps=[str(t) for t in toy_acc_2val.time_range],
            persist=False,
        )

        # Inject a ratio_eq row that legitimately bridges Mass<->CC.
        n_edges = toy_acc_2val.n_edges
        bad_row = sp.csr_matrix(
            ([1.0, -0.01], ([0, 0], [0, 5])), shape=(1, n_edges),
        )
        cr.G = sp.vstack([cr.G, bad_row], format="csr")
        cr.row_meta = pd.concat([
            cr.row_meta,
            pd.DataFrame([{
                "constraint_id": "fakeratio",
                "type":          "ratio",
                "is_hard":       False,
                "cone_type":     "free",
                "time":          "2020",
                "valuation":     None,
            }]),
        ], ignore_index=True)
        for td in cr.timestep_data.values():
            td["c"]       = np.append(td["c"], 0.0)
            td["sigma_c"] = np.append(td["sigma_c"], 1.0)

        # Must not raise.
        engine._check(cr)


# ---------------------------------------------------------------------------
# RatioConstraintBuilder
# ---------------------------------------------------------------------------

class TestRatioConstraintBuilder:

    def _write_params_csv(self, tmp_path, nominal=0.5, unc_type="sd", value_sd=0.05,
                          bound_lower="", bound_upper=""):
        import csv
        path = tmp_path / "params.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd",
                "value_min", "value_max",
                "bound_lower", "bound_upper",
            ])
            w.writeheader()
            w.writerow({
                "param_id":       "testratio",
                "a_sector_from":  "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to":    "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                "a_time":         "2020",   "a_valuation":   "Mass",
                "b_sector_from":  "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                "b_sector_to":    "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                "b_time":         "same",   "b_valuation":   "same",
                "value_nominal":  str(nominal),
                "unc_type":       unc_type,
                "value_sd":       str(value_sd),
                "value_min":      "",
                "value_max":      "",
                "bound_lower":    str(bound_lower),
                "bound_upper":    str(bound_upper),
            })
        return str(path)

    def test_ratio_soft_g_structure(self, toy_acc, tmp_path):
        """G row has +1 at edge_i and −r at edge_j; c=0."""
        from pysuto.constraints import RatioConstraintBuilder
        path    = self._write_params_csv(tmp_path, nominal=0.5, unc_type="sd", value_sd=0.05)
        builder = RatioConstraintBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)

        assert block.G_rows.shape[0] > 0, "Should produce at least 1 row"
        G = block.G_rows.toarray()
        row = G[0]
        # One entry should be +1, another −r
        assert 1.0 in row, "+1 coefficient not found"
        assert -0.5 in row, "-r coefficient not found"
        assert np.all(block.c == 0.0)

    def test_ratio_sigma_from_sd(self, toy_acc, tmp_path):
        """sigma_c == value_sd for unc_type='sd'."""
        from pysuto.constraints import RatioConstraintBuilder
        path    = self._write_params_csv(tmp_path, nominal=1.0, unc_type="sd", value_sd=0.123)
        builder = RatioConstraintBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        assert block.G_rows.shape[0] > 0
        assert block.sigma_c[0] == pytest.approx(0.123)

    def test_ratio_sigma_from_min_max(self, toy_acc, tmp_path):
        """sigma_c = (max - min) / 4 for unc_type='min_max'."""
        import csv
        from pysuto.constraints import RatioConstraintBuilder

        path = str(tmp_path / "p2.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd",
                "value_min", "value_max",
                "bound_lower", "bound_upper",
            ])
            w.writeheader()
            w.writerow({
                "param_id":       "mmratio",
                "a_sector_from":  "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to":    "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                "a_time":         "2020",   "a_valuation":   "Mass",
                "b_sector_from":  "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                "b_sector_to":    "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                "b_time":         "same",   "b_valuation":   "same",
                "value_nominal":  "1.0",
                "unc_type":       "min_max",
                "value_sd":       "",
                "value_min":      "0.8",
                "value_max":      "1.2",
                "bound_lower":    "",
                "bound_upper":    "",
            })

        builder = RatioConstraintBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        assert block.G_rows.shape[0] > 0
        expected = (1.2 - 0.8) / 4.0
        assert block.sigma_c[0] == pytest.approx(expected)

    def test_ratio_cone_type(self, toy_acc, tmp_path):
        """Ratio constraint rows have cone_type='free' and is_hard=False."""
        from pysuto.constraints import RatioConstraintBuilder
        path    = self._write_params_csv(tmp_path)
        builder = RatioConstraintBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        if block.G_rows.shape[0] > 0:
            assert (block.row_meta["cone_type"] == "free").all()
            assert (~block.row_meta["is_hard"]).all()

    def _write_agg_csv(self, tmp_path):
        """One agg row: pool every use-edge into Mining (a-side) against the
        Mining→Ore product (b-side), at Mass. Mirrors the eff pooled recipe."""
        import csv
        path = tmp_path / "agg.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id", "agg",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to", "a_time", "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to", "b_time", "b_valuation",
                "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
                "bound_lower", "bound_upper",
            ])
            w.writeheader()
            w.writerow({
                "param_id": "aggeff", "agg": "true",
                "a_sector_from": "*", "a_entity_from": "*", "a_region_from": "*",
                "a_sector_to": "Mining", "a_entity_to": "Process", "a_region_to": "*",
                "a_time": "*", "a_valuation": "Mass",
                "b_sector_from": "Mining", "b_entity_from": "Process", "b_region_from": "*",
                "b_sector_to": "Ore", "b_entity_to": "Flow", "b_region_to": "*",
                "b_time": "*", "b_valuation": "Mass",
                "value_nominal": "0.8", "unc_type": "sd", "value_sd": "0.05",
                "value_min": "", "value_max": "", "bound_lower": "", "bound_upper": "",
            })
        return str(path)

    def test_ratio_agg_buckets_per_region_and_year(self, toy_acc, tmp_path):
        """An agg row emits one set-scoped soft row per (region_to, time, valuation):
        Σb − r·Σa = 0. toy_acc has Mining in regions A,B over 2 years → 4 rows."""
        from pysuto.constraints import RatioConstraintBuilder
        builder = RatioConstraintBuilder.from_csv(self._write_agg_csv(tmp_path), toy_acc)
        block   = builder.build_static(toy_acc)
        rm = block.row_meta
        assert block.G_rows.shape[0] == 4                      # 2 regions × 2 years
        assert rm["constraint_id"].nunique() == 4              # unique per bucket
        # set-scoped: no single numerator/denominator edge
        assert rm["num_sector_from"].isna().all()
        assert rm["denom_sector_from"].isna().all()
        assert (rm["cone_type"] == "free").all() and (~rm["is_hard"]).all()
        assert np.allclose(block.sigma_c, 0.05)
        assert np.all(block.c == 0.0)
        # each row: +1 on the product edge, −0.8 on the pooled input edge
        G = block.G_rows.toarray()
        for row in G:
            assert np.isclose(row, 1.0).sum() == 1
            assert np.isclose(row, -0.8).sum() == 1

    def test_ratio_single_pair_mass_output_ratio(self, toy_acc, tmp_path):
        """Stripping-ratio shape: a single-pair Mass ratio between two Mining
        outputs (WR/Ore) → +1·Waste − r·Ore (no agg)."""
        import csv
        from pysuto.constraints import RatioConstraintBuilder
        path = str(tmp_path / "strip.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id", "param_type",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to", "a_time", "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to", "b_time", "b_valuation",
                "value_nominal", "unc_type", "value_sd",
            ])
            w.writeheader()
            w.writerow({
                "param_id": "strippingratio", "param_type": "stripping_ratio",
                "a_sector_from": "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to": "Ore", "a_entity_to": "Flow", "a_region_to": "A",
                "a_time": "*", "a_valuation": "Mass",
                "b_sector_from": "Mining", "b_entity_from": "Process", "b_region_from": "same",
                "b_sector_to": "Waste", "b_entity_to": "SB", "b_region_to": "same",
                "b_time": "same", "b_valuation": "same",
                "value_nominal": "2.0", "unc_type": "sd", "value_sd": "0.5",
            })
        builder = RatioConstraintBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        assert block.G_rows.shape[0] >= 1
        G = block.G_rows.toarray()
        row = G[0]
        assert 1.0 in row                # +1 on WR (numerator)
        assert -2.0 in row               # −2 on Ore (denominator)
        assert (block.row_meta["num_sector_from"] == "Mining").all()   # pair-scoped

    def test_ratio_sigma_c_scaled_symmetrically(self, toy_acc, tmp_path):
        """build_dynamic rescales σ_r to flow space:
        σ_c = max(σ_r, rel_floor)·max(Σ|f_num|, r·Σ|f_denom|, abs_floor)."""
        from pysuto.constraints import RatioConstraintBuilder
        path    = self._write_params_csv(tmp_path, nominal=0.5, unc_type="sd", value_sd=0.05)
        builder = RatioConstraintBuilder.from_csv(
            path, toy_acc, sigma_rel_floor=0.02, scale_abs_floor=1.0)
        builder.build_static(toy_acc)
        p = builder._pairs[0]
        f = np.zeros(toy_acc.n_edges)
        # numerator-dominant: |f_num| > r·|f_denom|
        f[p["edge_i"]] = 1000.0
        f[p["edge_j"]] = 100.0
        dyn = builder.build_dynamic(toy_acc, f, None, p["time"])
        assert dyn.sigma_c[0] == pytest.approx(0.05 * 1000.0)
        # denominator-dominant: r·|f_denom| > |f_num|
        f[p["edge_i"]] = 10.0
        f[p["edge_j"]] = 4000.0
        dyn = builder.build_dynamic(toy_acc, f, None, p["time"])
        assert dyn.sigma_c[0] == pytest.approx(0.05 * 0.5 * 4000.0)

    def test_ratio_sigma_c_floors(self, toy_acc, tmp_path):
        """rel floor lifts a quasi-hard σ_r; abs floor keeps both-zero-prior
        rows at a finite σ_c."""
        from pysuto.constraints import RatioConstraintBuilder
        path    = self._write_params_csv(tmp_path, nominal=0.5, unc_type="sd", value_sd=0.001)
        builder = RatioConstraintBuilder.from_csv(
            path, toy_acc, sigma_rel_floor=0.02, scale_abs_floor=1.0)
        builder.build_static(toy_acc)
        p = builder._pairs[0]
        f = np.zeros(toy_acc.n_edges)
        # rel floor: σ_r=0.001 < 0.02 → 0.02 used
        f[p["edge_i"]] = 1000.0
        dyn = builder.build_dynamic(toy_acc, f, None, p["time"])
        assert dyn.sigma_c[0] == pytest.approx(0.02 * 1000.0)
        # abs floor: both priors zero → scale = 1.0
        f[:] = 0.0
        dyn = builder.build_dynamic(toy_acc, f, None, p["time"])
        assert dyn.sigma_c[0] == pytest.approx(0.02 * 1.0)

    def test_ratio_sigma_c_scaled_agg(self, toy_acc, tmp_path):
        """Agg (pooled) rows scale by the pooled sums Σ|f| over each side."""
        from pysuto.constraints import RatioConstraintBuilder
        builder = RatioConstraintBuilder.from_csv(
            self._write_agg_csv(tmp_path), toy_acc,
            sigma_rel_floor=0.02, scale_abs_floor=1.0)
        builder.build_static(toy_acc)
        p = builder._pairs[0]
        f = np.zeros(toy_acc.n_edges)
        f[p["edge_i_list"]] = 100.0
        f[p["edge_j_list"]] = 300.0
        dyn = builder.build_dynamic(toy_acc, f, None, p["time"])
        expected = max(0.05, 0.02) * max(
            100.0 * len(p["edge_i_list"]),
            p["r"] * 300.0 * len(p["edge_j_list"]),
            1.0,
        )
        assert dyn.sigma_c[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RatioBoundBuilder
# ---------------------------------------------------------------------------

class TestRatioBoundBuilder:

    def _write_bound_csv(self, tmp_path, bu=2.0, bl=0.1):
        import csv
        path = tmp_path / "bound_params.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd",
                "value_min", "value_max",
                "bound_lower", "bound_upper",
            ])
            w.writeheader()
            w.writerow({
                "param_id":       "boundtest",
                "a_sector_from":  "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to":    "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                "a_time":         "2020",   "a_valuation":   "Mass",
                "b_sector_from":  "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                "b_sector_to":    "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                "b_time":         "same",   "b_valuation":   "same",
                "value_nominal":  "1.0",
                "unc_type":       "sd",
                "value_sd":       "0.1",
                # RatioBoundBuilder reads the band from value_min/value_max
                # (bound_lower/bound_upper are legacy/ignored — see conc params).
                "value_min":      str(bl),
                "value_max":      str(bu),
                "bound_lower":    "",
                "bound_upper":    "",
            })
        return str(path)

    def test_ratio_hard_two_rows_per_param(self, toy_acc, tmp_path):
        """Each param row with both bounds produces exactly 2 G rows."""
        from pysuto.constraints import RatioBoundBuilder
        path    = self._write_bound_csv(tmp_path)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        assert block.G_rows.shape[0] == 2, \
            f"Expected 2 rows (upper + lower); got {block.G_rows.shape[0]}"

    def test_ratio_bound_inequality_direction(self, toy_acc, tmp_path):
        """Upper row: G[i]=+1, G[j]=−bu. Lower row: G[i]=−1, G[j]=+bl.

        Asserts coefficients per column (not just value membership): the
        bound coefficient must sit on the denominator edge (a-side) and the
        ±1 on the numerator edge (b-side). Under the solver convention
        G·x ≤ 0 this encodes bl·denom ≤ num ≤ bu·denom; the swapped
        placement instead pins denom ≤ bu⁻¹·num (regression: mass collapse
        in opt_20260611_075036_6d2e540f).
        """
        from pysuto.constraints import RatioBoundBuilder
        path    = self._write_bound_csv(tmp_path, bu=2.0, bl=0.1)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        G       = block.G_rows.toarray()

        # _write_bound_csv: a-edge (denominator) = Mining/A→Ore/A 2020 = eid 0,
        #                   b-edge (numerator)   = Ore/A→Mining/A 2020 = eid 2.
        denom_col, num_col = 0, 2
        upper_row = G[0]
        lower_row = G[1]
        assert upper_row[num_col] == pytest.approx(1.0) and \
               upper_row[denom_col] == pytest.approx(-2.0), \
            f"Upper row should be +1 on num (col {num_col}), -bu on denom " \
            f"(col {denom_col}); got {upper_row[upper_row != 0]}"
        assert lower_row[num_col] == pytest.approx(-1.0) and \
               lower_row[denom_col] == pytest.approx(0.1), \
            f"Lower row should be -1 on num (col {num_col}), +bl on denom " \
            f"(col {denom_col}); got {lower_row[lower_row != 0]}"

    def test_ratio_bound_cone_type(self, toy_acc, tmp_path):
        """All ratio bound rows have cone_type='nonneg' and is_hard=True."""
        from pysuto.constraints import RatioBoundBuilder
        path    = self._write_bound_csv(tmp_path)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        assert (block.row_meta["cone_type"] == "nonneg").all()
        assert block.row_meta["is_hard"].all()

    def test_ratio_bound_sigma_zero(self, toy_acc, tmp_path):
        """Hard bound constraints have sigma_c = 0."""
        from pysuto.constraints import RatioBoundBuilder
        path    = self._write_bound_csv(tmp_path)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        assert np.all(block.sigma_c == 0.0)

    # ── empty-band guard (regression: scrap dual-class collapse) ──

    def _write_two_bound_csv(self, tmp_path, b1, b2):
        """Two bound params on the SAME (a,b) edge pair, with bands b1, b2."""
        import csv
        path = tmp_path / "two_bound_params.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
            ])
            w.writeheader()
            for pid, (bl, bu) in (("classlo", b1), ("classhi", b2)):
                w.writerow({
                    "param_id":      pid,
                    "a_sector_from": "Mining", "a_entity_from": "Process", "a_region_from": "A",
                    "a_sector_to":   "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                    "a_time":        "2020",   "a_valuation":   "Mass",
                    "b_sector_from": "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                    "b_sector_to":   "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                    "b_time":        "same",   "b_valuation":   "same",
                    "value_nominal": str(0.5 * (bl + bu)), "unc_type": "min_max",
                    "value_sd":      "", "value_min": str(bl), "value_max": str(bu),
                })
        return str(path)

    def test_empty_band_disjoint_params_raises(self, toy_acc, tmp_path):
        """Two hard bounds on one edge-pair whose bands don't overlap → raise.

        Mirrors the scrap class-1 [0.9,0.98] ∩ class-2 [0.4,0.6] = ∅ collapse:
        effective band [max(lo), min(hi)] is empty, so the flow is pinned to 0.
        """
        from pysuto.constraints import RatioBoundBuilder
        from pysuto.constraints.engine import _validate_ratio_bounds
        path  = self._write_two_bound_csv(tmp_path, (0.4, 0.6), (0.9, 0.98))
        block = RatioBoundBuilder.from_csv(path, toy_acc).build_static(toy_acc)
        with pytest.raises(ValueError, match="empty interval"):
            _validate_ratio_bounds(block.G_rows, block.row_meta)

    def test_overlapping_bands_ok(self, toy_acc, tmp_path):
        """A single wide band [0.4,0.98] (post-merge scrap) is non-empty → OK."""
        from pysuto.constraints import RatioBoundBuilder
        from pysuto.constraints.engine import _validate_ratio_bounds
        path  = self._write_bound_csv(tmp_path, bu=0.98, bl=0.4)
        block = RatioBoundBuilder.from_csv(path, toy_acc).build_static(toy_acc)
        _validate_ratio_bounds(block.G_rows, block.row_meta)  # must not raise

    def test_value_min_gt_max_raises(self, toy_acc, tmp_path):
        """A single param with value_min > value_max is rejected at load time."""
        from pysuto.constraints import RatioBoundBuilder
        path = self._write_bound_csv(tmp_path, bu=0.4, bl=0.9)  # bl>bu
        with pytest.raises(ValueError, match="empty ratio band"):
            RatioBoundBuilder.from_csv(path, toy_acc)

    # ── net-stock-change bound orientation (regression: cathode-pool collapse) ──

    def test_stock_bound_agg_orientation(self, toy_acc, tmp_path):
        """An agg stock-shaped bound encodes `stock ≤ p95·denom`, not its inverse.

        The stock change (b-side, here Mining→Waste/SB) is the numerator and must
        carry the ±1; the denominator (a-side, Mining→Ore) carries ∓bu/bl — the
        `Σb − r·Σa` form (CONTEXT.md §"Node-scoped vs set-scoped"; ADR 0005). The
        swapped placement builds `denom ≤ p95·stock` → forces stock ≥ denom/p95
        (~20–42×), the cathode-pool collapse this guards against.
        """
        import csv
        from pysuto.constraints import RatioBoundBuilder
        path = tmp_path / "stock_bound.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id", "agg",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to", "a_entity_to", "a_region_to", "a_time", "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to", "b_entity_to", "b_region_to", "b_time", "b_valuation",
                "value_nominal", "value_min", "value_max",
            ])
            w.writeheader()
            w.writerow({
                "param_id": "netstockboundtest", "agg": "true",
                # a-side = DENOMINATOR (Mining→Ore)
                "a_sector_from": "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to": "Ore", "a_entity_to": "Flow", "a_region_to": "A",
                "a_time": "2020", "a_valuation": "Mass",
                # b-side = NUMERATOR / stock change (Mining→Waste)
                "b_sector_from": "Mining", "b_entity_from": "Process", "b_region_from": "A",
                "b_sector_to": "Waste", "b_entity_to": "SB", "b_region_to": "A",
                "b_time": "2020", "b_valuation": "Mass",
                "value_nominal": "0", "value_min": "-0.05", "value_max": "0.05",
            })
        block = RatioBoundBuilder.from_csv(str(path), toy_acc).build_static(toy_acc)
        assert block.G_rows.shape[0] == 2, "one (region,time,val) bucket × (upper, lower)"
        G = block.G_rows.toarray()
        stock_col, denom_col = 4, 0   # eid 4 = Mining/A→Waste/A 2020; eid 0 = Mining/A→Ore/A 2020
        upper, lower = G[0], G[1]
        assert upper[stock_col] == pytest.approx(1.0) and upper[denom_col] == pytest.approx(-0.05), \
            f"upper must be +1 on stock (col {stock_col}), -bu on denom (col {denom_col}); got {upper[upper != 0]}"
        assert lower[stock_col] == pytest.approx(-1.0) and lower[denom_col] == pytest.approx(-0.05), \
            f"lower must be -1 on stock, +bl(=-0.05) on denom; got {lower[lower != 0]}"

    def _load_stock_bounds_generator(self):
        """Import scripts/explo/stock_bounds_study.py by path (it is import-safe:
        data reads live in functions, main() is guarded)."""
        import importlib.util, pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        mod_path = repo / "scripts" / "explo" / "stock_bounds_study.py"
        try:
            spec = importlib.util.spec_from_file_location("stock_bounds_study", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ImportError as e:                       # e.g. matplotlib absent in a minimal env
            pytest.skip(f"stock_bounds_study not importable: {e}")
        return mod

    @pytest.mark.parametrize("indicator,stock,denom", [
        ("PSC", "PSC", "Cathode"),
        ("USC", "USC", "FU"),
        ("MESC", "MESC", "FU"),
    ])
    def test_stock_bound_generator_orientation(self, indicator, stock, denom):
        """`_build_param_row` writes the stock edge on the b-side (numerator) and the
        FU/Cathode denominator on the a-side — the orientation RatioBoundBuilder needs.
        A straight a→a / b→b mapping inverts the bound (cathode-pool collapse)."""
        gen = self._load_stock_bounds_generator()
        row = gen._build_param_row(indicator, "Chile", 0.05, "test", "denom-label")
        assert row["b_sector_to"] == stock, f"{indicator}: stock edge must be the b-side numerator"
        assert row["a_sector_to"] == denom, f"{indicator}: denominator must be the a-side"
        assert float(row["value_min"]) < 0 < float(row["value_max"]), "two-sided ±p95 band in value_min/max"

    def test_stock_bound_params_contract(self):
        """The shipped stock_bounds params keep the correct orientation: stock edge
        (PSC/USC/MESC) on the b-side, FU/Cathode on the a-side, symmetric ±band in
        value_min/value_max. Skips when the (gitignored) params file is absent."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        p = repo / "data" / "input" / "params" / "stock_bounds" / "current" / "params.csv"
        if not p.exists():
            pytest.skip("stock_bounds/current/params.csv absent (gitignored data dir)")
        df = pd.read_csv(p, sep=None, engine="python", dtype=str)
        s = df[df["param_type"] == "stock_bound"]
        assert len(s) > 0, "no stock_bound rows in shipped params"
        assert s["b_sector_to"].isin(["PSC", "USC", "MESC"]).all(), "stock edge must be the b-side numerator"
        assert s["a_sector_to"].isin(["Cathode", "FU"]).all(), "denominator must be the a-side"
        vmin = pd.to_numeric(s["value_min"]); vmax = pd.to_numeric(s["value_max"])
        assert (vmin < 0).all() and (vmax > 0).all() and np.allclose(vmin, -vmax), \
            "value_min/value_max must be a symmetric ±p95 band"


# ---------------------------------------------------------------------------
# FlowBoundsBuilder
# ---------------------------------------------------------------------------

class TestFlowBoundsBuilder:

    def test_flow_bounds_shape(self, toy_acc):
        """u and l from acc_struct have shape (E,)."""
        assert toy_acc.bounds.u.shape == (toy_acc.n_edges,)
        assert toy_acc.bounds.l.shape == (toy_acc.n_edges,)

    def test_flow_bounds_no_g_rows(self, toy_acc):
        """FlowBoundsBuilder produces no G rows; bounds live in ConstraintRun.u/.l."""
        from pysuto.constraints import FlowBoundsBuilder
        block = FlowBoundsBuilder().build_static(toy_acc)
        assert block.G_rows.shape[0] == 0, \
            f"Expected 0 G rows (bounds stored separately); got {block.G_rows.shape[0]}"

    def test_flow_bounds_carried_in_run(self, toy_acc, toy_agg_result):
        """ConstraintEngine.run() carries non-default bounds in cr.u / cr.l."""
        from pysuto.constraints import ConstraintEngine, FlowBoundsBuilder
        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_fbb_run")
        engine.register(FlowBoundsBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)
        # toy_acc has edge 0 with upper=100
        assert run.u[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# ConstraintEngine
# ---------------------------------------------------------------------------

class TestConstraintEngine:

    def test_engine_register_and_run_shapes(self, toy_acc, toy_agg_result):
        """G.shape=(M, E), c.shape=(M,), sigma_c.shape=(M,); u/l.shape=(E,)."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_engine")
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        M, E = run.G.shape
        assert E == toy_acc.n_edges
        assert M > 0

        for label, td in run.timestep_data.items():
            assert td["c"].shape       == (M,)
            assert td["sigma_c"].shape == (M,)

        assert run.u.shape == (toy_acc.n_edges,)
        assert run.l.shape == (toy_acc.n_edges,)

    def test_g_is_time_invariant(self, toy_acc, toy_agg_result):
        """G matrix is identical for two different timesteps."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_tiv")
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020", "2021"], persist=False)

        # G is stored once; c(t) may differ — but G itself is time-invariant by design
        assert run.G is not None
        G_dense = run.G.toarray()
        # Both timesteps reference the same G object
        assert run.timestep_data["2020"]["c"].shape == (run.G.shape[0],)
        assert run.timestep_data["2021"]["c"].shape == (run.G.shape[0],)

    def test_row_meta_completeness(self, toy_acc, toy_agg_result):
        """row_meta has one row per G row; required columns present."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_meta")
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        required = {"constraint_id", "type", "is_hard", "cone_type"}
        assert required.issubset(run.row_meta.columns)
        assert len(run.row_meta) == run.G.shape[0]

    def test_engine_no_builders(self, toy_acc, toy_agg_result):
        """Engine with no builders produces an empty G but valid ConstraintRun."""
        from pysuto.constraints import ConstraintEngine

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_empty")
        run    = engine.run(toy_agg_result, timesteps=["2020"], persist=False)
        assert run.G.shape == (0, toy_acc.n_edges)

    def test_engine_multiple_builders(self, toy_acc, toy_agg_result):
        """Multiple registered builders produce correct total row count."""
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder, RatioBoundBuilder,
        )
        import csv, os
        csv_path = "/tmp/test_multi_bound.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd",
                "value_min", "value_max", "bound_lower", "bound_upper",
            ])
            w.writeheader()
            w.writerow({
                "param_id": "mb",
                "a_sector_from": "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to":   "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                "a_time": "2020", "a_valuation": "Mass",
                "b_sector_from": "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                "b_sector_to":   "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                "b_time": "same", "b_valuation": "same",
                "value_nominal": "1.0", "unc_type": "sd", "value_sd": "0.1",
                "value_min": "0.5", "value_max": "2.0", "bound_lower": "", "bound_upper": "",
            })

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_multi")
        engine.register(MassBalanceBuilder())
        engine.register(RatioBoundBuilder.from_csv(csv_path, toy_acc))
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        assert run.G.shape[0] > 0
        assert "massbal" in run.row_meta["type"].values
        assert "ratio"   in run.row_meta["type"].values
        # param_id "mb" becomes the variant; the bound rows are the nonneg cone.
        assert "mb"     in run.row_meta["variant"].values
        assert "nonneg" in run.row_meta["cone_type"].values


# ---------------------------------------------------------------------------
# ConstraintRunRegistry
# ---------------------------------------------------------------------------

class TestConstraintRunRegistry:

    def test_registry_round_trip(self, toy_acc, toy_agg_result, tmp_path):
        """Save ConstraintRun; reload; G and row_meta match."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
        from pysuto.registry import ConstraintRunRegistry
        storage = str(tmp_path / "const")
        engine  = ConstraintEngine(toy_acc, storage_dir=storage)
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=True)

        registry  = ConstraintRunRegistry(storage_dir=storage)
        run2      = registry.load(run.run_id)

        assert run2.G.shape       == run.G.shape
        assert len(run2.row_meta) == len(run.row_meta)
        assert np.allclose(run2.a0,      run.a0)
        assert np.allclose(run2.sigma_a, run.sigma_a)
        assert np.allclose(run2.u,       run.u)
        assert np.allclose(run2.l,       run.l)
        td1 = run.timestep_data["2020"]
        td2 = run2.timestep_data["2020"]
        assert np.allclose(td1["c"],       td2["c"])
        assert np.allclose(td1["sigma_c"], td2["sigma_c"])

    def test_registry_summary_df(self, toy_acc, toy_agg_result, tmp_path):
        """summary_df() returns DataFrame with one row per registered run."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
        from pysuto.registry import ConstraintRunRegistry
        storage = str(tmp_path / "const2")
        for _ in range(2):
            engine = ConstraintEngine(toy_acc, storage_dir=storage)
            engine.register(MassBalanceBuilder())
            engine.run(toy_agg_result, timesteps=["2020"], persist=True)

        registry = ConstraintRunRegistry(storage_dir=storage)
        df       = registry.summary_df()
        assert len(df) == 2
        assert "run_id" in df.columns

    def test_load_unknown_run_id_raises(self, tmp_path):
        """Loading a non-existent run_id raises KeyError."""
        from pysuto.registry import ConstraintRunRegistry
        reg = ConstraintRunRegistry(storage_dir=str(tmp_path / "const"))
        with pytest.raises(KeyError):
            reg.load("nonexistent_run")


# ---------------------------------------------------------------------------
# Per-timestep deactivation of ratio_eq / ratio_bound (Fix 8.2)
# ---------------------------------------------------------------------------

class TestRatioPerTimestepDeactivation:
    """build_dynamic should flip σ_c (ratio_eq) / c (ratio_bound) for off-year
    rows so per-t solves only see the in-year ratio constraints."""

    def _write_wildcard_params_csv(self, tmp_path, name="params.csv",
                                   nominal=0.5, bl="", bu=""):
        import csv
        path = tmp_path / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd",
                "value_min", "value_max",
                "bound_lower", "bound_upper",
            ])
            w.writeheader()
            w.writerow({
                "param_id":       "test",
                "a_sector_from":  "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to":    "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                "a_time":         "*",      "a_valuation":   "Mass",
                "b_sector_from":  "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                "b_sector_to":    "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                "b_time":         "same",   "b_valuation":   "same",
                "value_nominal":  str(nominal),
                "unc_type":       "sd",
                "value_sd":       "0.05",
                # RatioBoundBuilder reads the band from value_min/value_max.
                "value_min":      str(bl),
                "value_max":      str(bu),
                "bound_lower":    "",
                "bound_upper":    "",
            })
        return str(path)

    def test_ratio_eq_sigma_c_inactive_for_off_year(self, toy_acc, tmp_path):
        """build_dynamic at t=2020 sets σ_c=SIGMA_C_INACTIVE for 2021 rows."""
        from pysuto.constraints import RatioConstraintBuilder, SIGMA_C_INACTIVE
        path    = self._write_wildcard_params_csv(tmp_path)
        builder = RatioConstraintBuilder.from_csv(path, toy_acc)
        static  = builder.build_static(toy_acc)
        # Sanity: wildcard expands to 2 rows (2020, 2021).
        assert static.G_rows.shape[0] == 2
        times = static.row_meta["time"].astype(str).to_numpy()
        assert set(times) == {"2020", "2021"}

        # f_agg = 1 everywhere → flow-space scale = max(1, r·1, abs_floor=1) = 1,
        # so the active row's σ_c equals the static σ_r.
        dyn = builder.build_dynamic(toy_acc, np.ones(toy_acc.n_edges), None, "2020")
        for i, t in enumerate(times):
            if t == "2020":
                assert dyn.sigma_c[i] == static.sigma_c[i]
            else:
                assert dyn.sigma_c[i] == SIGMA_C_INACTIVE
        # c always 0 for ratio_eq; G_rows passed through.
        assert np.all(dyn.c == 0.0)
        assert dyn.G_rows.shape == static.G_rows.shape

    def test_ratio_bound_c_inactive_for_off_year(self, toy_acc, tmp_path):
        """build_dynamic at t=2020 sets c=BOUND_C_INACTIVE for 2021 rows."""
        from pysuto.constraints import RatioBoundBuilder, BOUND_C_INACTIVE
        path    = self._write_wildcard_params_csv(tmp_path, bl=0.1, bu=2.0)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        static  = builder.build_static(toy_acc)
        # Wildcard expansion: 2 (years) × 2 (upper+lower) = 4 rows.
        assert static.G_rows.shape[0] == 4
        times = static.row_meta["time"].astype(str).to_numpy()

        dyn = builder.build_dynamic(toy_acc, None, None, "2020")
        for i, t in enumerate(times):
            if t == "2020":
                assert dyn.c[i] == 0.0
            else:
                assert dyn.c[i] == BOUND_C_INACTIVE
        # σ_c stays 0 (hard rows); G_rows passed through.
        assert np.all(dyn.sigma_c == 0.0)
        assert dyn.G_rows.shape == static.G_rows.shape

    def test_ratio_bound_sentinel_trivially_satisfies_inequality(self, toy_acc, tmp_path):
        """For x at u_max=1e9, A x ≤ BOUND_C_INACTIVE element-wise on inactive rows."""
        from pysuto.constraints import RatioBoundBuilder, BOUND_C_INACTIVE
        path    = self._write_wildcard_params_csv(tmp_path, bl=0.1, bu=2.0)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        builder.build_static(toy_acc)
        dyn = builder.build_dynamic(toy_acc, None, None, "2020")

        x   = np.full(toy_acc.n_edges, 1e9)
        Ax  = dyn.G_rows @ x
        times = dyn.row_meta["time"].astype(str).to_numpy()
        inactive = times != "2020"
        # On inactive rows, Ax ≤ BOUND_C_INACTIVE must hold.
        assert np.all(Ax[inactive] <= dyn.c[inactive])

    def test_engine_per_t_dispatch_uses_dynamic(self, toy_acc, toy_agg_result, tmp_path):
        """ConstraintEngine.run wires build_dynamic per timestep — verify the
        per-t sigma_c / c arrays differ between 2020 and 2021 solves."""
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder,
            RatioConstraintBuilder, RatioBoundBuilder,
            SIGMA_C_INACTIVE, BOUND_C_INACTIVE,
        )
        engine = ConstraintEngine(toy_acc, storage_dir=str(tmp_path / "const"))
        engine.register(MassBalanceBuilder())
        engine.register(RatioConstraintBuilder.from_csv(
            self._write_wildcard_params_csv(tmp_path, name="re.csv"), toy_acc,
        ))
        engine.register(RatioBoundBuilder.from_csv(
            self._write_wildcard_params_csv(tmp_path, name="rb.csv", bl=0.1, bu=2.0),
            toy_acc,
        ))
        run = engine.run(
            toy_agg_result, timesteps=["2020", "2021"], persist=False,
        )
        rm   = run.row_meta
        t2020 = run.timestep_data["2020"]
        t2021 = run.timestep_data["2021"]

        re_mask = (rm["type"] == "ratio_eq").to_numpy()
        rb_mask = (rm["type"] == "ratio_bound").to_numpy()
        row_t   = rm["time"].astype(str).to_numpy()

        # ratio_eq: 2021 rows deactivated at t=2020. (σ_c may be Lp-perturbed
        # by engine post-processing — assert close to the sentinel, not equal.)
        assert np.allclose(
            t2020["sigma_c"][re_mask & (row_t == "2021")], SIGMA_C_INACTIVE,
            rtol=1e-6,
        )
        # ratio_eq: 2020 rows active at t=2020 (σ_c finite, far below sentinel).
        assert np.all(t2020["sigma_c"][re_mask & (row_t == "2020")] < 1.0)

        # ratio_bound: 2021 rows deactivated at t=2020 via c=BOUND_C_INACTIVE.
        # c is not regularised, so exact equality is fine here.
        assert np.all(t2020["c"][rb_mask & (row_t == "2021")] == BOUND_C_INACTIVE)
        assert np.all(t2020["c"][rb_mask & (row_t == "2020")] == 0.0)
        # Symmetric at t=2021.
        assert np.all(t2021["c"][rb_mask & (row_t == "2020")] == BOUND_C_INACTIVE)
        assert np.all(t2021["c"][rb_mask & (row_t == "2021")] == 0.0)

    def test_engine_check_warns_on_non_uniform_ratio_rows(self, toy_acc, toy_agg_result,
                                                          tmp_path, caplog):
        """Per-year uniformity check fires WARN when a year has fewer rows."""
        import logging
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder, RatioConstraintBuilder,
        )
        # CSV with hardcoded a_time=2020 → only 1 ratio_eq row, not uniform
        # across the 2-year toy_acc.time_range.
        path = tmp_path / "single_year.csv"
        TestRatioConstraintBuilder()._write_params_csv  # type: ignore[attr-defined]
        # Reuse the existing single-year CSV builder via the existing class.
        path_str = TestRatioConstraintBuilder()._write_params_csv(
            tmp_path, nominal=0.5,
        )
        engine = ConstraintEngine(toy_acc, storage_dir=str(tmp_path / "const"))
        engine.register(MassBalanceBuilder())
        engine.register(RatioConstraintBuilder.from_csv(path_str, toy_acc))
        with caplog.at_level(logging.WARNING):
            engine.run(
                toy_agg_result, timesteps=["2020", "2021"], persist=False,
            )
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "ratio_eq" in m and ("non-uniform" in m or "not evenly divisible" in m)
            for m in warn_msgs
        ), f"Expected non-uniform ratio_eq warning. Got: {warn_msgs}"


# ---------------------------------------------------------------------------
# Per-timestep G partition (Slice 2 of the per-t block plan)
# ---------------------------------------------------------------------------

class TestEnginePerTPartition:
    """ConstraintEngine.run exposes per-t G blocks so per-t solves can act on
    a (rows_for_t, edges_for_t) submatrix instead of the full (M, E) matrix."""

    def test_g_blocks_action_matches_legacy_G(self, toy_acc, toy_agg_result, tmp_path):
        """For each t, G_blocks[t] @ x[edge_cols_by_t[t]] equals the slice of
        legacy_G @ x at rows tagged with t. Partition is lossless."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
        engine = ConstraintEngine(toy_acc, storage_dir=str(tmp_path / "const"))
        engine.register(MassBalanceBuilder())
        cr = engine.run(toy_agg_result, timesteps=["2020", "2021"], persist=False)

        assert set(cr.G_blocks.keys()) == {"2020", "2021"}

        rng = np.random.default_rng(0)
        x   = rng.standard_normal(toy_acc.n_edges)
        legacy_full = cr.G @ x
        times = cr.row_meta["time"].astype(str).to_numpy()

        for t, G_t in cr.G_blocks.items():
            cols_t        = cr.edge_cols_by_t[t]
            block_action  = G_t @ x[cols_t]
            legacy_action = legacy_full[times == t]
            assert np.allclose(block_action, legacy_action), \
                f"Mismatch at t={t}: per-block action != legacy slice"

    def test_g_blocks_survive_registry_roundtrip(self, toy_acc, toy_agg_result, tmp_path):
        """Slice 8: registry save→load preserves G_blocks + edge_cols_by_t."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
        from pysuto.registry   import ConstraintRunRegistry
        storage = str(tmp_path / "const")
        engine = ConstraintEngine(toy_acc, storage_dir=storage)
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020", "2021"], persist=True)

        registry = ConstraintRunRegistry(storage_dir=storage)
        loaded   = registry.load(run.run_id)

        assert set(loaded.G_blocks.keys()) == set(run.G_blocks.keys())
        for t in run.G_blocks:
            diff = (loaded.G_blocks[t] - run.G_blocks[t])
            assert diff.nnz == 0, f"G_blocks[{t}] differs after roundtrip"
            np.testing.assert_array_equal(
                loaded.edge_cols_by_t[t], run.edge_cols_by_t[t],
            )

    def test_g_blocks_action_with_mass_balance_and_ratios(
        self, toy_acc, toy_agg_result, tmp_path,
    ):
        """Partition stays lossless with a mixed builder set (mass_balance +
        ratio_eq + ratio_bound). Exercises rows of every cone_type."""
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder,
            RatioConstraintBuilder, RatioBoundBuilder,
        )
        # Wildcard params: time-invariant ratio + bound across t=2020, 2021.
        params_csv = TestRatioPerTimestepDeactivation()._write_wildcard_params_csv(
            tmp_path, nominal=0.5, bl=0.1, bu=2.0,
        )
        engine = ConstraintEngine(toy_acc, storage_dir=str(tmp_path / "const"))
        engine.register(MassBalanceBuilder())
        engine.register(RatioConstraintBuilder.from_csv(params_csv, toy_acc))
        engine.register(RatioBoundBuilder.from_csv(params_csv, toy_acc))
        cr = engine.run(toy_agg_result, timesteps=["2020", "2021"], persist=False)

        rng   = np.random.default_rng(1)
        x     = rng.standard_normal(toy_acc.n_edges)
        times = cr.row_meta["time"].astype(str).to_numpy()
        legacy_full = cr.G @ x

        for t, G_t in cr.G_blocks.items():
            block_action  = G_t @ x[cr.edge_cols_by_t[t]]
            legacy_action = legacy_full[times == t]
            assert np.allclose(block_action, legacy_action), \
                f"Mismatch at t={t} (mixed builders)"

        # Rows of every family/cone survive the partition: massbal (zero),
        # ratio_eq (free) and ratio_bound (nonneg). The shared param_id is the
        # variant, so eq vs bound is told apart by cone_type, not variant.
        types_seen = set()
        cones_seen = set()
        for t in cr.G_blocks:
            t_rm = cr.row_meta.loc[times == t]
            types_seen |= set(t_rm["type"].unique())
            cones_seen |= set(t_rm["cone_type"].unique())
        assert {"massbal", "ratio"} <= types_seen
        assert {"zero", "free", "nonneg"} <= cones_seen


class TestPriorBox:
    """`_apply_prior_box` grafts the RAS scale-floor onto the QP bounds."""

    def _engine(self, k):
        from pysuto.constraints import ConstraintEngine
        e = ConstraintEngine.__new__(ConstraintEngine)
        e._prior_box_k = float(k) if k else None
        return e

    def test_off_is_noop(self):
        e = self._engine(None)
        a0 = np.array([100.0, 0.0, 10.0])
        sa = np.array([7.0, 1.0, 2.0])
        l = np.array([0.0, 0.0, 0.0])
        u = np.array([1e9, 1e9, 1e9])
        ln, un = e._apply_prior_box(l.copy(), u.copy(), a0, sa)
        assert np.array_equal(ln, l) and np.array_equal(un, u)

    def test_boxes_priored_edges(self):
        e = self._engine(2.0)
        a0 = np.array([100.0, 1000.0])
        sa = np.array([7.0, 50.0])
        l = np.array([0.0, 0.0])
        u = np.array([1e9, 1e9])
        ln, un = e._apply_prior_box(l.copy(), u.copy(), a0, sa)
        assert np.allclose(ln, [86.0, 900.0])
        assert np.allclose(un, [114.0, 1100.0])

    def test_flat_prior_untouched(self):
        # a0 == 0 (FlatPrior) keeps its original [l, u] — not pinned to zero.
        e = self._engine(2.0)
        a0 = np.array([0.0])
        sa = np.array([1.0])
        l = np.array([0.0])
        u = np.array([1e9])
        ln, un = e._apply_prior_box(l.copy(), u.copy(), a0, sa)
        assert ln[0] == 0.0 and un[0] == 1e9

    def test_single_edge_conflict_reverts(self):
        # a0 < 0 with a sign bound l=0 would invert (u < l); that edge reverts
        # so only cross-constraint infeasibility surfaces, not single-edge empties.
        e = self._engine(2.0)
        a0 = np.array([-5.0])
        sa = np.array([1.0])
        l = np.array([0.0])
        u = np.array([1e9])
        ln, un = e._apply_prior_box(l.copy(), u.copy(), a0, sa)
        assert ln[0] == 0.0 and un[0] == 1e9

    def test_never_widens(self):
        e = self._engine(10.0)
        a0 = np.array([100.0])
        sa = np.array([50.0])           # a0 - kσ = -400, a0 + kσ = 600
        l = np.array([10.0])            # tighter than the box low side
        u = np.array([200.0])           # tighter than the box high side
        ln, un = e._apply_prior_box(l.copy(), u.copy(), a0, sa)
        assert ln[0] == 10.0 and un[0] == 200.0


class TestConstraintIdGrammar:
    """The {type}_{variant}_{specs}_{year} grammar and uniform row_meta schema
    across a full multi-builder engine run."""

    def _full_run(self, toy_acc, toy_agg_result, tmp_path):
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder,
            SumConstraintBuilder, RatioConstraintBuilder, RatioBoundBuilder,
        )
        params = TestRatioPerTimestepDeactivation()._write_wildcard_params_csv(
            tmp_path, nominal=0.5, bl=0.1, bu=2.0,
        )
        engine = ConstraintEngine(toy_acc, storage_dir=str(tmp_path / "const"))
        engine.register(MassBalanceBuilder())
        engine.register(SumConstraintBuilder.from_acc_struct(toy_acc, sigma_fraction=0.05))
        engine.register(RatioConstraintBuilder.from_csv(params, toy_acc))
        engine.register(RatioBoundBuilder.from_csv(params, toy_acc))
        return engine.run(toy_agg_result, timesteps=["2020", "2021"], persist=False)

    def test_row_meta_schema_is_uniform(self, toy_acc, toy_agg_result, tmp_path):
        from pysuto.constraints.naming import ROW_META_COLS
        cr = self._full_run(toy_acc, toy_agg_result, tmp_path)
        assert list(cr.row_meta.columns) == ROW_META_COLS

    def test_ids_unique(self, toy_acc, toy_agg_result, tmp_path):
        cr = self._full_run(toy_acc, toy_agg_result, tmp_path)
        assert cr.row_meta["constraint_id"].is_unique

    def test_id_grammar_round_trips(self, toy_acc, toy_agg_result, tmp_path):
        cr = self._full_run(toy_acc, toy_agg_result, tmp_path)
        rm = cr.row_meta
        for cid, typ, variant, time in zip(
            rm["constraint_id"], rm["type"], rm["variant"], rm["time"],
        ):
            toks = str(cid).split("_")
            assert len(toks) >= 4, cid
            assert toks[0] == typ, (cid, typ)            # slot 1 == type column
            assert toks[1] == variant, (cid, variant)    # slot 2 == variant
            assert toks[-1] == (str(time) if time is not None else "na")
        assert set(rm["type"].unique()) <= {"massbal", "sum", "ratio"}

    def test_ratio_struct_cols_populated_only_on_ratio(self, toy_acc, toy_agg_result, tmp_path):
        cr = self._full_run(toy_acc, toy_agg_result, tmp_path)
        rm = cr.row_meta
        is_ratio = rm["type"] == "ratio"
        # ratio rows carry the numerator/denominator decode; others do not.
        assert rm.loc[is_ratio, "num_sector_from"].notna().all()
        assert rm.loc[is_ratio, "denom_valuation"].notna().all()
        assert rm.loc[~is_ratio, "num_sector_from"].isna().all()


# ---------------------------------------------------------------------------
# Supply/use total-output SCALE TOTALS (self-anchored to f_agg). See ADR 0006.
# ---------------------------------------------------------------------------

def _make_fd_acc():
    """Minimal acc: one flow (Cathode/A) consumed by a process (use) AND by
    final demand (Cathode→FU/SB), to exercise use-side summation over all
    destinations including final demand."""
    from pysuto.acc import AccountingStructure
    import logging
    acc = AccountingStructure.__new__(AccountingStructure)
    acc.logger = logging.getLogger("test_fd_acc")
    acc.id = "toyfd"
    acc.node_index = pd.DataFrame(
        [
            ("Smelter", "Process", "A"),   # 0
            ("Cathode", "Flow",    "A"),   # 1
            ("Fab",     "Process", "A"),   # 2
            ("FU",      "SB",      "A"),   # 3  final-demand boundary
        ],
        columns=["sector", "entity", "region"],
    )
    acc.n_nodes    = 4
    acc.time_range = ["2020"]
    acc.valuation  = ["Mass"]
    acc.edge_index = pd.DataFrame([
        {"edge_id": 0, "from_id": 0, "to_id": 1, "time": "2020", "valuation": "Mass"},  # supply Smelter→Cathode
        {"edge_id": 1, "from_id": 1, "to_id": 2, "time": "2020", "valuation": "Mass"},  # use    Cathode→Fab
        {"edge_id": 2, "from_id": 1, "to_id": 3, "time": "2020", "valuation": "Mass"},  # FD     Cathode→FU (SB)
    ])
    acc.n_edges = 3
    return acc


class TestSupplyUseScaleTotal:
    """Supply/use total-output scale totals, self-anchored to f_agg, grouped by
    (sector_from, region_from, time, valuation). Mirrors the wiring in
    scripts/constraints.py. See ADR 0006."""

    GROUP_KEYS = ("sector_from", "region_from", "time", "valuation")
    CV = 0.01

    @staticmethod
    def _masks(acc):
        from pysuto.param_relation import expand_edge_index
        ei = expand_edge_index(acc.edge_index, acc.node_index)
        supply = (ei["entity_from"].eq("Process") & ei["entity_to"].eq("Flow")).to_numpy()
        use    = ei["entity_from"].eq("Flow").to_numpy()
        return supply, use

    def test_supply_scale_total_one_row_per_process(self, toy_acc, toy_agg_result):
        """Supply side: one node-scoped row per process; each process→flow edge
        IS the total supply (no summing); c = f_agg[edge]; σ_c = cv·|c|."""
        from pysuto.constraints import SumConstraintBuilder
        supply, _ = self._masks(toy_acc)
        f_agg = toy_agg_result.f_agg
        block = SumConstraintBuilder.from_mask_anchor(
            toy_acc, mask=supply, anchor_values=f_agg,
            group_keys=self.GROUP_KEYS, sigma_cv=self.CV, id_prefix="supplytotout",
        ).build_static(toy_acc)
        rm = block.row_meta

        # supply edges {0,1,5,6} → one (Mining, region, year) group each = 4 rows
        assert block.G_rows.shape[0] == 4
        assert (rm["type"] == "sum").all()
        assert (rm["variant"] == "supplytotout").all()
        assert (rm["cone_type"] == "free").all()
        assert (~rm["is_hard"].astype(bool)).all()
        # node identity populated (re-grill: sector filled, symmetric with region)
        assert (rm["sector"] == "Mining").all()
        assert set(rm["region"]) == {"A", "B"}

        G = block.G_rows.toarray()
        for i, c in enumerate(block.c):
            eids = np.flatnonzero(G[i] != 0.0)
            assert len(eids) == 1                      # no summing on the supply side
            assert np.isclose(G[i, eids[0]], 1.0)
            assert np.isclose(c, f_agg[eids[0]])
            assert np.isclose(block.sigma_c[i], self.CV * abs(c))

    def test_use_scale_total_one_row_per_flow(self, toy_acc, toy_agg_result):
        """Use side: one node-scoped row per flow; c = Σ f_agg over its
        outgoing edges."""
        from pysuto.constraints import SumConstraintBuilder
        _, use = self._masks(toy_acc)
        f_agg = toy_agg_result.f_agg
        block = SumConstraintBuilder.from_mask_anchor(
            toy_acc, mask=use, anchor_values=f_agg,
            group_keys=self.GROUP_KEYS, sigma_cv=self.CV, id_prefix="usetotout",
        ).build_static(toy_acc)
        rm = block.row_meta

        assert block.G_rows.shape[0] == 4
        assert (rm["variant"] == "usetotout").all()
        assert (rm["sector"] == "Ore").all()
        G = block.G_rows.toarray()
        for i, c in enumerate(block.c):
            eids = np.flatnonzero(G[i] != 0.0)
            assert np.allclose(G[i, eids], 1.0)
            assert np.isclose(c, f_agg[eids].sum())

    def test_use_scale_total_sums_destinations_incl_final_demand(self):
        """A flow used by a process AND by final demand → one row whose c sums
        BOTH outgoing edges (use + flow→SB final demand)."""
        from pysuto.constraints import SumConstraintBuilder
        from pysuto.param_relation import expand_edge_index
        acc   = _make_fd_acc()
        f_agg = np.array([10.0, 6.0, 4.0])     # supply, use, final demand
        ei    = expand_edge_index(acc.edge_index, acc.node_index)
        use_mask = ei["entity_from"].eq("Flow").to_numpy()   # {1,2}: Cathode→Fab + Cathode→FU

        block = SumConstraintBuilder.from_mask_anchor(
            acc, mask=use_mask, anchor_values=f_agg,
            group_keys=self.GROUP_KEYS, sigma_cv=self.CV, id_prefix="usetotout",
        ).build_static(acc)

        assert block.G_rows.shape[0] == 1          # one flow node
        G = block.G_rows.toarray()
        eids = np.flatnonzero(G[0] != 0.0)
        assert set(eids.tolist()) == {1, 2}        # use AND final demand included
        assert np.allclose(G[0, eids], 1.0)
        assert np.isclose(block.c[0], f_agg[1] + f_agg[2])   # 6 + 4 = 10
        assert np.isclose(block.sigma_c[0], self.CV * abs(block.c[0]))
        assert block.row_meta["sector"].iloc[0] == "Cathode"
        assert block.row_meta["variant"].iloc[0] == "usetotout"

    def test_scale_total_zero_group_excluded(self, toy_acc, toy_agg_result):
        """A group whose f_agg sum is ~0 is dropped (0-exclusion)."""
        from pysuto.constraints import SumConstraintBuilder
        supply, _ = self._masks(toy_acc)
        f_agg = toy_agg_result.f_agg.copy()
        f_agg[6] = 0.0                              # supply edge for (Mining, B, 2021)
        block = SumConstraintBuilder.from_mask_anchor(
            toy_acc, mask=supply, anchor_values=f_agg,
            group_keys=self.GROUP_KEYS, sigma_cv=self.CV, id_prefix="supplytotout",
        ).build_static(toy_acc)
        rm = block.row_meta
        assert block.G_rows.shape[0] == 3          # the zero group is skipped
        present = set(zip(rm["region"].astype(str), rm["time"].astype(str)))
        assert ("B", "2021") not in present
        assert present == {("A", "2020"), ("B", "2020"), ("A", "2021")}

    def test_scale_total_per_year_deactivation(self, toy_acc, toy_agg_result):
        """Rows are year-locked: build_dynamic at 2020 deactivates 2021 rows."""
        from pysuto.constraints import SumConstraintBuilder, SIGMA_C_INACTIVE
        supply, _ = self._masks(toy_acc)
        builder = SumConstraintBuilder.from_mask_anchor(
            toy_acc, mask=supply, anchor_values=toy_agg_result.f_agg,
            group_keys=self.GROUP_KEYS, sigma_cv=self.CV, id_prefix="supplytotout",
        )
        block = builder.build_static(toy_acc)
        rm    = block.row_meta
        zeros = np.zeros(toy_acc.n_edges)
        b20   = builder.build_dynamic(toy_acc, zeros, zeros, "2020")
        is2021 = (rm["time"].astype(str) == "2021").to_numpy()
        assert np.allclose(b20.sigma_c[is2021], SIGMA_C_INACTIVE)
        assert np.allclose(b20.sigma_c[~is2021], block.sigma_c[~is2021])

    def test_scale_total_both_valuations(self, toy_acc_2val):
        """Both valuations yield unit-coherent rows, one per (process, region,
        year, valuation)."""
        from pysuto.constraints import SumConstraintBuilder
        supply, _ = self._masks(toy_acc_2val)
        f_agg = np.arange(1.0, toy_acc_2val.n_edges + 1.0)   # all > 0
        block = SumConstraintBuilder.from_mask_anchor(
            toy_acc_2val, mask=supply, anchor_values=f_agg,
            group_keys=self.GROUP_KEYS, sigma_cv=self.CV, id_prefix="supplytotout",
        ).build_static(toy_acc_2val)
        rm = block.row_meta
        # Mining×{A,B} × {2020,2021} × {Mass,CC} = 8 rows
        assert block.G_rows.shape[0] == 8
        assert set(rm["valuation"]) == {"Mass", "CC"}
        assert (rm["sector"] == "Mining").all()


# ---------------------------------------------------------------------------
# No-data ("flat-prior") σ_a magnitude anchor
# ---------------------------------------------------------------------------

def _make_anchor_acc():
    """Minimal acc for the no-data σ_a anchor: a metallurgical sector (Scrap,
    not excluded) with one anchored sibling + one no-data edge, and an
    extraction edge (Mining, excluded)."""
    from pysuto.acc import AccountingStructure
    acc = AccountingStructure.__new__(AccountingStructure)
    acc.node_index = pd.DataFrame(
        [
            ("Scrap",    "SB",      "A"),   # 0
            ("Smelting", "Process", "A"),   # 1
            ("Mining",   "Process", "A"),   # 2
            ("Ore",      "Flow",    "A"),   # 3
        ],
        columns=["sector", "entity", "region"],
    )
    acc.edge_index = pd.DataFrame([
        {"edge_id": 0, "from_id": 0, "to_id": 1, "time": "2020", "valuation": "Mass"},  # Scrap→Smelting anchored
        {"edge_id": 1, "from_id": 0, "to_id": 1, "time": "2020", "valuation": "Mass"},  # Scrap→Smelting no-data
        {"edge_id": 2, "from_id": 2, "to_id": 3, "time": "2020", "valuation": "Mass"},  # Mining→Ore no-data (excluded)
    ])
    acc.n_edges = len(acc.edge_index)
    return acc


class TestNodataSigmaAnchor:

    def test_anchors_nodata_metallurgical_edge(self):
        """A no-data edge (a₀=0, σ=1e6) in a non-excluded sector gets
        σ = cv·median|a₀| of anchored siblings; the anchored sibling and the
        extraction edge are left untouched."""
        from pysuto.constraints import ConstraintEngine
        acc = _make_anchor_acc()
        eng = ConstraintEngine(acc_struct=acc)            # cv default 0.3
        a0      = np.array([100.0, 0.0,  0.0])
        sigma_a = np.array([  5.0, 1e6,  1e6])
        out = eng._anchor_nodata_sigma(a0, sigma_a, acc)
        assert out[1] == pytest.approx(0.3 * 100.0)        # cv · median(anchored Scrap a₀)
        assert out[0] == 5.0                               # anchored sibling untouched
        assert out[2] == 1e6                               # Mining excluded → still flat
        assert sigma_a[1] == 1e6                           # original not mutated in place

    def test_disable_switch(self):
        """nodata_anchor_cv=0 leaves every σ_a untouched."""
        from pysuto.constraints import ConstraintEngine
        acc = _make_anchor_acc()
        eng = ConstraintEngine(acc_struct=acc, nodata_anchor_cv=0.0)
        a0      = np.array([100.0, 0.0, 0.0])
        sigma_a = np.array([  5.0, 1e6, 1e6])
        out = eng._anchor_nodata_sigma(a0, sigma_a, acc)
        np.testing.assert_array_equal(out, sigma_a)

    def test_below_threshold_not_anchored(self):
        """A zero-prior edge whose σ is already below the flat threshold (i.e.
        not a global-median fallback) is left alone — only the 1e6 fallback is
        retargeted."""
        from pysuto.constraints import ConstraintEngine
        acc = _make_anchor_acc()
        eng = ConstraintEngine(acc_struct=acc)
        a0      = np.array([100.0, 0.0, 0.0])
        sigma_a = np.array([  5.0, 50.0, 1e6])             # e1 σ below 1e5 threshold
        out = eng._anchor_nodata_sigma(a0, sigma_a, acc)
        assert out[1] == 50.0                              # untouched (not a fallback σ)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])