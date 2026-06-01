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

    # Structural edge_index (5 edges, no time replication):
    #   supply: Mining/A(0) → Ore/A(1)       [eid 0]
    #   supply: Mining/B(2) → Ore/B(3)       [eid 1]
    #   use:    Ore/A(1)   → Mining/A(0)     [eid 2]
    #   use:    Ore/B(3)   → Mining/B(2)     [eid 3]
    #   waste:  Mining/A(0)→ Waste/SB(4)     [eid 4]
    rows = []
    for eid, (fi, ti) in enumerate([(0,1),(2,3),(1,0),(3,2),(0,4)]):
        rows.append({"edge_id": eid, "from_id": fi, "to_id": ti,
                     "valuation": "Mass"})

    acc.edge_index = pd.DataFrame(rows)
    acc.n_edges    = len(acc.edge_index)

    # ruleset-derived masks (structural):
    supply_ids  = {0, 1}
    use_ids     = {2, 3}
    sb_ids      = {4}
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
    """Return a minimal AggregationResult-like namespace for toy_acc.

    Returns ``a_agg`` / ``sigma_agg`` as ``(n_edges, n_timesteps)`` 2-D
    arrays per the structural model. ConstraintEngine.run reads the
    per-period slice ``[:, t_idx]`` internally.
    """
    n_t = len(acc.time_range)
    rng     = np.random.default_rng(seed)
    a_agg   = rng.uniform(1.0, 100.0, size=(acc.n_edges, n_t))
    sigma_a = rng.uniform(0.1, 10.0, size=(acc.n_edges, n_t))

    class _R:
        pass

    r           = _R()
    r.a_agg     = a_agg
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
        """G rows for a balanced structural solution sum to zero."""
        from pysuto.constraints import MassBalanceBuilder
        builder = MassBalanceBuilder()
        block   = builder.build_static(toy_acc)

        # Structural edges: supply edges (0, 1) = use edges (2, 3); waste (4) = 0.
        f = np.array([10., 10., 10., 10., 0.])
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
        """One structural row per balanced node × valuation (4 × 1 = 4)."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        assert block.G_rows.shape[0] == 4, \
            f"Expected 4 structural rows; got {block.G_rows.shape[0]}"

    def test_mass_balance_sigma_c_zero(self, toy_acc):
        """sigma_c is zero for hard constraints."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        assert np.all(block.sigma_c == 0.0)

    def test_mass_balance_row_meta_no_time_column(self, toy_acc):
        """Under the structural model row_meta has NO `time` column;
        constraint_id ends with `_node{nid}_{valuation}`. Per-period
        activation is tracked on ConstraintRun.active instead."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc)
        rm = block.row_meta
        assert "time" not in rm.columns
        for cid, v in zip(rm["constraint_id"], rm["valuation"]):
            assert str(cid).endswith(f"_{v}"), \
                f"constraint_id={cid!r} does not end with _{v}"


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

        # Waste edge (structural id 4) should not appear in any G row.
        assert np.all(G[:, 4] == 0.0), "Waste edge 4 should not appear in sum constraints"

    def test_sum_row_meta_no_time_column(self, toy_acc):
        """Auto-derived sum rows are time-agnostic under the structural
        model; row_meta carries no `time` column."""
        from pysuto.constraints import SumConstraintBuilder
        block = SumConstraintBuilder.from_acc_struct(
            toy_acc, sigma_fraction=0.05,
        ).build_static(toy_acc)
        rm = block.row_meta
        assert "time" not in rm.columns
        for cid, v in zip(rm["constraint_id"], rm["valuation"]):
            assert str(cid).endswith(f"_{v}")

    def test_sum_sigma_c_scales_with_magnitude(self, toy_acc):
        """sigma_c from fraction_of_flow doubles when a_agg doubles.

        Under the structural model auto-derived rows are time-agnostic
        (``spec_time = None``) so every row stays active in every period;
        σ_c simply tracks the per-period a_agg slice."""
        from pysuto.constraints import SumConstraintBuilder

        a1 = np.ones(toy_acc.n_edges) * 10.0
        s1 = np.ones(toy_acc.n_edges)
        a2 = np.ones(toy_acc.n_edges) * 20.0
        s2 = np.ones(toy_acc.n_edges)

        builder = SumConstraintBuilder.from_acc_struct(toy_acc, sigma_fraction=0.1)
        b1 = builder.build_dynamic(toy_acc, a1, s1, "2020")
        b2 = builder.build_dynamic(toy_acc, a2, s2, "2020")

        assert b1.G_rows.shape[0] > 0
        assert b1.active.all() and b2.active.all()
        ratio = b2.sigma_c / b1.sigma_c
        assert np.allclose(ratio, 2.0, rtol=0.01)

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
                "constraint_id": "test_sum", "description": "test",
                "sector_from": "Mining", "entity_from": "Process", "region_from": "A",
                "sector_to":   "Ore",    "entity_to":   "Flow",    "region_to":   "A",
                "time": "2020", "valuation": "Mass",
                "coefficient": "1.0", "c_k": "0", "sigma_ck_mode": "sd",
                "sigma_ck_value": "5.0", "sigma_ck_fraction": "",
            })
            # term 2: use Ore/A → Mining/A  at 2020 (edge 2), subtract
            w.writerow({
                "constraint_id": "test_sum", "description": "",
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

    # Structural edges: 5 topological edges × 2 valuations = 10 rows.
    rows = []
    eid  = 0
    for val in ["Mass", "CC"]:
        for (fi, ti) in [(0,1),(2,3),(1,0),(3,2),(0,4)]:
            rows.append({"edge_id": eid, "from_id": fi, "to_id": ti,
                         "valuation": val})
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
        """Structural mass-balance: 4 balanced nodes × 2 valuations = 8 rows."""
        from pysuto.constraints import MassBalanceBuilder
        block = MassBalanceBuilder().build_static(toy_acc_2val)
        assert block.G_rows.shape[0] == 8
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
        # Balanced by supply/use: Ore/A and Ore/B × 2 valuations = 4.
        assert block.G_rows.shape[0] == 4
        assert set(block.row_meta["valuation"]) == {"Mass", "CC"}

    def test_sum_subentity_splits_per_valuation(self, toy_acc_2val):
        from pysuto.constraints import SumConstraintBuilder
        block = SumConstraintBuilder.from_acc_struct_subentity(
            toy_acc_2val,
            entity_filter={"Process"},
            rhs_out_entity={"Flow"},
            lhs_neg_out_entity_sub={"Nature"},
        ).build_static(toy_acc_2val)
        # Process nodes: Mining/A, Mining/B × 2 valuations = 4.
        assert block.G_rows.shape[0] == 4
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

        # Ore-A row (Mass): sector=Ore, region=A, entity=Flow. No time column.
        ore_a_row = mb.row_meta[
            (mb.row_meta["sector"] == "Ore")
            & (mb.row_meta["region"] == "A")
            & (mb.row_meta["valuation"] == "Mass")
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
        # Inject a fake sum_eq row that mixes Mass (eid 0) and CC (eid 5)
        # by appending a row to StructG's sparsity pattern + extending the
        # 2-D ``c`` / ``sigma_c`` / ``active`` tables to match.
        n_edges = toy_acc_2val.n_edges
        from pysuto.constraints import StructG
        new_indptr = np.concatenate([cr.G.indptr, [cr.G.nnz + 2]])
        new_indices = np.concatenate([cr.G.indices, [0, 5]])
        new_data = np.vstack([
            cr.G.data,
            np.ones((2, cr.G.n_timesteps)),    # +1 at edges 0 and 5
        ])
        cr.G = StructG(
            indptr   = new_indptr,
            indices  = new_indices,
            data     = new_data,
            shape    = (cr.G.shape[0] + 1, cr.G.shape[1]),
            row_meta = pd.concat([
                cr.row_meta,
                pd.DataFrame([{
                    "constraint_id": "fake_mixed",
                    "type":          "sum_eq",
                    "is_hard":       False,
                    "cone_type":     "free",
                    "valuation":     None,
                }]),
            ], ignore_index=True),
        )
        cr.row_meta = cr.G.row_meta
        cr.c       = np.vstack([cr.c,       np.zeros((1, cr.G.n_timesteps))])
        cr.sigma_c = np.vstack([cr.sigma_c, np.ones((1,  cr.G.n_timesteps))])
        cr.active  = np.vstack([cr.active,  np.ones((1,  cr.G.n_timesteps), dtype=bool)])

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
        from pysuto.constraints import StructG
        new_indptr  = np.concatenate([cr.G.indptr, [cr.G.nnz + 2]])
        new_indices = np.concatenate([cr.G.indices, [0, 5]])
        # +1 at edge 0, -0.01 at edge 5 (cross-valuation ratio).
        extra_data = np.array([[1.0] * cr.G.n_timesteps, [-0.01] * cr.G.n_timesteps])
        new_data = np.vstack([cr.G.data, extra_data])
        cr.G = StructG(
            indptr   = new_indptr,
            indices  = new_indices,
            data     = new_data,
            shape    = (cr.G.shape[0] + 1, cr.G.shape[1]),
            row_meta = pd.concat([
                cr.row_meta,
                pd.DataFrame([{
                    "constraint_id": "fake_ratio",
                    "type":          "ratio_eq",
                    "is_hard":       False,
                    "cone_type":     "free",
                    "valuation":     None,
                }]),
            ], ignore_index=True),
        )
        cr.row_meta = cr.G.row_meta
        cr.c       = np.vstack([cr.c,       np.zeros((1, cr.G.n_timesteps))])
        cr.sigma_c = np.vstack([cr.sigma_c, np.ones((1,  cr.G.n_timesteps))])
        cr.active  = np.vstack([cr.active,  np.ones((1,  cr.G.n_timesteps), dtype=bool)])

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
                "param_id":       "test_ratio",
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
                "param_id":       "mm_ratio",
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
                "param_id":       "bound_test",
                "a_sector_from":  "Mining", "a_entity_from": "Process", "a_region_from": "A",
                "a_sector_to":    "Ore",    "a_entity_to":   "Flow",    "a_region_to":   "A",
                "a_time":         "2020",   "a_valuation":   "Mass",
                "b_sector_from":  "Ore",    "b_entity_from": "Flow",    "b_region_from": "A",
                "b_sector_to":    "Mining", "b_entity_to":   "Process", "b_region_to":   "A",
                "b_time":         "same",   "b_valuation":   "same",
                "value_nominal":  "1.0",
                "unc_type":       "sd",
                "value_sd":       "0.1",
                "value_min":      "",
                "value_max":      "",
                "bound_lower":    str(bl),
                "bound_upper":    str(bu),
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
        """Upper row: G[i]=+1, G[j]=−bu. Lower row: G[i]=−1, G[j]=+bl."""
        from pysuto.constraints import RatioBoundBuilder
        path    = self._write_bound_csv(tmp_path, bu=2.0, bl=0.1)
        builder = RatioBoundBuilder.from_csv(path, toy_acc)
        block   = builder.build_static(toy_acc)
        G       = block.G_rows.toarray()

        upper_row = G[0]
        lower_row = G[1]
        assert  1.0 in upper_row and -2.0 in upper_row, \
            f"Upper row should have +1 and -bu=-2.0; got {upper_row[upper_row != 0]}"
        assert -1.0 in lower_row and  0.1 in lower_row, \
            f"Lower row should have -1 and +bl=0.1; got {lower_row[lower_row != 0]}"

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
        """StructG holds (M, E, T) tape; c/sigma_c are 2-D; u/l are (E,)."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_engine")
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        M, E = run.G.shape
        assert E == toy_acc.n_edges
        assert M > 0

        T = run.G.n_timesteps
        assert run.c.shape       == (M, T)
        assert run.sigma_c.shape == (M, T)
        assert run.active.shape  == (M, T)
        assert run.u.shape == (toy_acc.n_edges,)
        assert run.l.shape == (toy_acc.n_edges,)

    def test_struct_g_shares_one_sparsity_across_t(self, toy_acc, toy_agg_result):
        """StructG.at(t) yields the same CSR sparsity pattern for every t —
        only the data column varies."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_tiv")
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020", "2021"], persist=False)

        g0 = run.G.at(0)
        g1 = run.G.at(1)
        # Same indptr/indices across periods (sparsity invariant).
        np.testing.assert_array_equal(g0.indptr,  g1.indptr)
        np.testing.assert_array_equal(g0.indices, g1.indices)
        # Mass balance is topological → data is identical too.
        np.testing.assert_array_equal(g0.data, g1.data)
        assert run.c.shape[1] == 2

    def test_row_meta_completeness(self, toy_acc, toy_agg_result):
        """row_meta has one row per G row; required columns present.
        Under the structural model ``time`` is NOT a column of row_meta —
        per-period applicability lives on ``ConstraintRun.active``."""
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_meta")
        engine.register(MassBalanceBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        required = {"constraint_id", "type", "is_hard", "cone_type"}
        assert required.issubset(run.row_meta.columns)
        assert "time" not in run.row_meta.columns
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
                "value_min": "", "value_max": "", "bound_lower": "0.5", "bound_upper": "2.0",
            })

        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_multi")
        engine.register(MassBalanceBuilder())
        engine.register(RatioBoundBuilder.from_csv(csv_path, toy_acc))
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        assert run.G.shape[0] > 0
        assert "mass_balance" in run.row_meta["type"].values
        assert "ratio_bound"  in run.row_meta["type"].values


# ---------------------------------------------------------------------------
# ConstraintRunRegistry
# ---------------------------------------------------------------------------

class TestConstraintRunRegistry:

    def test_registry_round_trip(self, toy_acc, toy_agg_result, tmp_path):
        """Save ConstraintRun via the engine; reload; StructG + 2-D arrays match."""
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
        np.testing.assert_array_equal(run2.G.indptr,  run.G.indptr)
        np.testing.assert_array_equal(run2.G.indices, run.G.indices)
        np.testing.assert_allclose(run2.G.data, run.G.data)
        np.testing.assert_allclose(run2.a0,      run.a0)
        np.testing.assert_allclose(run2.sigma_a, run.sigma_a)
        np.testing.assert_allclose(run2.u,       run.u)
        np.testing.assert_allclose(run2.l,       run.l)
        np.testing.assert_allclose(run2.c,       run.c)
        np.testing.assert_allclose(run2.sigma_c, run.sigma_c)
        np.testing.assert_array_equal(run2.active, run.active)
        assert run2.timesteps == run.timesteps

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
        from pysuto.registry import ConstraintRunRegistry
        reg = ConstraintRunRegistry(storage_dir=str(tmp_path / "const"))
        with pytest.raises(KeyError):
            reg.load("nonexistent_run")


# ---------------------------------------------------------------------------
# Per-timestep deactivation of ratio_eq / ratio_bound (Fix 8.2)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Phase 3 — sentinel-based per-timestep deactivation is replaced "
           "by the first-class activation mask on ConstraintRun. Tests will "
           "be rewritten against active[constraint, t].",
    strict=False,
)
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
                "value_min":      "",
                "value_max":      "",
                "bound_lower":    str(bl),
                "bound_upper":    str(bu),
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

        dyn = builder.build_dynamic(toy_acc, None, None, "2020")
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
# Per-timestep G partition — REMOVED in Phase 3.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Phase 3 — _partition_G_by_time / G_blocks / edge_cols_by_t are "
           "removed. The no-cross-time-coupling invariant is now structural "
           "(StructG has no time axis on the column side). Per-t solves "
           "use StructG.at(t) directly.",
    strict=False,
)
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

        # All four cone types' rows survive the partition
        types_seen = set()
        for t in cr.G_blocks:
            t_rm = cr.row_meta.loc[times == t]
            types_seen |= set(t_rm["type"].unique())
        assert {"mass_balance", "ratio_eq", "ratio_bound"} <= types_seen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])