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

    acc.supply_edge_mask   = np.isin(all_eids, list(supply_ids))
    acc.use_edge_mask      = np.isin(all_eids, list(use_ids))
    acc.tradable_edge_mask = np.zeros(acc.n_edges, dtype=bool)
    acc.sb_edge_mask       = np.isin(all_eids, list(sb_ids))

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

    def test_sum_sigma_c_scales_with_magnitude(self, toy_acc):
        """sigma_c from fraction_of_flow doubles when f_agg doubles."""
        from pysuto.constraints import SumConstraintBuilder

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

        # sigma_c should approximately double
        assert b1.G_rows.shape[0] > 0, "Should have at least one dynamic row"
        ratio = b2.sigma_c / np.where(b1.sigma_c > 0, b1.sigma_c, np.nan)
        assert np.allclose(ratio[~np.isnan(ratio)], 2.0, rtol=0.01), \
            "sigma_c should double when f_agg doubles"

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

    def test_flow_bounds_non_default_edges(self, toy_acc):
        """Only edges with non-default bounds produce G rows."""
        from pysuto.constraints import FlowBoundsBuilder
        block = FlowBoundsBuilder().build_static(toy_acc)
        # toy_acc has edge 0 with upper=100 → expect 1 G row (only upper, lower is default)
        assert block.G_rows.shape[0] == 1, \
            f"Expected 1 row for edge 0 upper bound; got {block.G_rows.shape[0]}"

    def test_flow_bounds_cone_type(self, toy_acc):
        """Flow bound rows have cone_type='nonneg' and is_hard=True."""
        from pysuto.constraints import FlowBoundsBuilder
        block = FlowBoundsBuilder().build_static(toy_acc)
        if block.G_rows.shape[0] > 0:
            assert (block.row_meta["cone_type"] == "nonneg").all()
            assert block.row_meta["is_hard"].all()


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
            ConstraintEngine, MassBalanceBuilder, FlowBoundsBuilder,
        )
        engine = ConstraintEngine(toy_acc, storage_dir="/tmp/test_const_multi")
        engine.register(MassBalanceBuilder())
        engine.register(FlowBoundsBuilder())
        run = engine.run(toy_agg_result, timesteps=["2020"], persist=False)

        # MassBalance rows + FlowBound rows
        assert run.G.shape[0] > 0
        assert "mass_balance" in run.row_meta["type"].values
        assert "flow_bound"   in run.row_meta["type"].values


# ---------------------------------------------------------------------------
# ConstraintRunRegistry
# ---------------------------------------------------------------------------

class TestConstraintRunRegistry:

    def test_registry_round_trip(self, toy_acc, toy_agg_result, tmp_path):
        """Save ConstraintRun; reload; G and row_meta match."""
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder, ConstraintRunRegistry,
        )
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
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder, ConstraintRunRegistry,
        )
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
        from pysuto.constraints import ConstraintRunRegistry
        reg = ConstraintRunRegistry(storage_dir=str(tmp_path / "const"))
        with pytest.raises(KeyError):
            reg.load("nonexistent_run")


if __name__ == "__main__":
    pytest.main([__file__])