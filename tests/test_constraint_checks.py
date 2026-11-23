"""
test_constraint_checks.py
=========================
The build-time checks of Stage 3 (#283, unit audit U3): what the engine and the builders
refuse, and the pieces of the unit that no other test file reaches (the p_mean σ_c and its
ordering behind σ_a, the flat-prior pin's hand-off, the soft-ratio pedigree mode, the
per-pair net-stock skip of #98, the const diagnostic's summary and figures).

Fixtures are the toy accounting structures of ``test_constraints.py``.
"""

from __future__ import annotations

import csv
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from test_constraints import _make_toy_acc, _make_toy_agg_result, _make_two_valuation_acc

from pysuto.constraints import (
    BaseConstraintBuilder,
    ConstraintBlock,
    ConstraintEngine,
    FlowBoundsBuilder,
    MassBalanceBuilder,
    RatioBoundBuilder,
    RatioConstraintBuilder,
    SumConstraintBuilder,
)
from pysuto.constraints.engine import _partition_G_by_time
from pysuto.constraints.naming import empty_row_meta_dict


def _engine(acc, tmp_path, **kw):
    args = dict(sigma_a_cv_min=0.0, sigma_a_abs_floor=0.0, sigma_c_cv_min=0.0,
                sigma_c_abs_floor=0.0, nodata_anchor_cv=0.0)
    args.update(kw)
    return ConstraintEngine(acc, storage_dir=str(tmp_path), **args)


class _StubBuilder(BaseConstraintBuilder):
    """One row on edge 0 in 2020: hard or soft, with a static σ_c and an optional
    dynamic overlay (a σ_c per year, or a block of ``dyn_n`` rows)."""

    def __init__(self, *, hard, sigma_c=0.0, dyn_sigma=None, dyn_n=None):
        self.hard, self.sigma_c = hard, sigma_c
        self.dyn_sigma, self.dyn_n = dyn_sigma or {}, dyn_n

    def _block(self, acc, n, sigma):
        meta = pd.DataFrame([{**empty_row_meta_dict(), "constraint_id": f"stub{i}",
                              "type": "sum", "variant": "stub", "is_hard": self.hard,
                              "sense": "eq", "time": "2020", "valuation": "Mass"}
                             for i in range(n)])
        G = sp.csr_matrix((np.ones(n), (np.arange(n), np.zeros(n, dtype=np.int64))),
                          shape=(n, acc.n_edges))
        return ConstraintBlock(G, np.zeros(n), np.full(n, float(sigma)), meta)

    def build_static(self, acc_struct):
        return self._block(acc_struct, 1, self.sigma_c)

    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t):
        if self.dyn_n is not None:
            return self._block(acc_struct, self.dyn_n, 1.0)
        if t in self.dyn_sigma:
            return self._block(acc_struct, 1, self.dyn_sigma[t])
        return super().build_dynamic(acc_struct, f_agg, sigma_agg, t)


YEARS = ["2020", "2021"]


# ---------------------------------------------------------------------------
# ConstraintEngine: the checks raise (U3-G4-2, U3-S2-1, U3-S2-3)
# ---------------------------------------------------------------------------

class TestEngineChecksRaise:

    def _run(self, acc, tmp_path, builder, **kw):
        agg = _make_toy_agg_result(acc)
        eng = _engine(acc, tmp_path, **kw)
        eng.register(builder)
        return eng.run(agg, timesteps=YEARS, persist=False)

    def test_clean_build_passes(self, tmp_path):
        cr = self._run(_make_toy_acc(), tmp_path, _StubBuilder(hard=False, sigma_c=1.0))
        assert cr.G.shape[0] == 1

    def test_soft_row_with_zero_sigma_c_raises(self, tmp_path):
        with pytest.raises(ValueError, match="soft row.*σ_c <= 0"):
            self._run(_make_toy_acc(), tmp_path, _StubBuilder(hard=False, sigma_c=0.0))

    def test_soft_row_with_zero_sigma_c_in_a_later_year_raises(self, tmp_path):
        stub = _StubBuilder(hard=False, sigma_c=1.0, dyn_sigma={"2021": 0.0})
        with pytest.raises(ValueError, match="t=2021"):
            self._run(_make_toy_acc(), tmp_path, stub)

    def test_hard_row_with_sigma_c_in_a_later_year_raises(self, tmp_path):
        stub = _StubBuilder(hard=True, sigma_c=0.0, dyn_sigma={"2021": 2.0})
        with pytest.raises(ValueError, match="hard row.*σ_c != 0 at t=2021"):
            self._run(_make_toy_acc(), tmp_path, stub)

    def test_u_below_l_raises(self, tmp_path):
        acc = _make_toy_acc()
        acc.bounds.l[3], acc.bounds.u[3] = 5.0, 1.0
        with pytest.raises(ValueError, match="1 edges have u < l"):
            self._run(acc, tmp_path, MassBalanceBuilder())

    def test_dynamic_block_of_another_row_count_raises(self, tmp_path):
        stub = _StubBuilder(hard=False, sigma_c=1.0, dyn_n=2)
        with pytest.raises(ValueError, match="matched to the static rows by position"):
            self._run(_make_toy_acc(), tmp_path, stub)

    def test_g_of_the_wrong_width_raises(self, tmp_path):
        acc = _make_toy_acc()
        cr = self._run(acc, tmp_path, MassBalanceBuilder())
        cr.G = sp.hstack([cr.G, sp.csr_matrix((cr.G.shape[0], 1))], format="csr")
        with pytest.raises(ValueError, match=r"G.shape\[1\]"):
            _engine(acc, tmp_path)._check(cr)


# ---------------------------------------------------------------------------
# ConstraintEngine: inputs (U3-G6-2, U3-S1-2)
# ---------------------------------------------------------------------------

class TestEngineInputs:

    def test_nan_prior_raises(self, tmp_path):
        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        agg.f_agg[2] = np.nan
        with pytest.raises(ValueError, match="agg_result contains NaN"):
            _engine(acc, tmp_path).run(agg, timesteps=YEARS, persist=False)

    def test_missing_flat_prior_raises_when_the_anchor_needs_it(self, tmp_path):
        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        agg.flat_prior = None
        with pytest.raises(ValueError, match="carries no flat_prior"):
            _engine(acc, tmp_path, nodata_anchor_cv=0.3).run(agg, timesteps=YEARS,
                                                              persist=False)

    def test_misshaped_flat_prior_raises(self, tmp_path):
        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        agg.flat_prior = np.zeros(acc.n_edges - 1, dtype=bool)
        with pytest.raises(ValueError, match="flat_prior has shape"):
            _engine(acc, tmp_path, nodata_anchor_cv=0.3).run(agg, timesteps=YEARS,
                                                              persist=False)

    def test_partition_rejects_a_nonzero_in_another_years_column(self):
        acc = _make_toy_acc()
        G = sp.csr_matrix(([1.0, 1.0], ([0, 0], [0, 5])), shape=(1, acc.n_edges))
        row_meta = pd.DataFrame({"constraint_id": ["r0"], "time": ["2020"]})
        with pytest.raises(ValueError, match="no cross-time coupling"):
            _partition_G_by_time(G, row_meta, acc.edge_index, YEARS)


class TestFlatPriorPin:
    """ADR 0013 am. 1: the pin sets l = u = 0 on ConstraintRun (the feasibility
    probe's hand-off), only on flat-prior edges outside the exclusion set."""

    def test_pin_sets_l_and_u_to_zero_on_flat_edges_only(self, tmp_path):
        acc = _make_toy_acc()
        acc.bounds.l[:] = -1.0
        agg = _make_toy_agg_result(acc)
        agg.flat_prior = np.zeros(acc.n_edges, dtype=bool)
        agg.flat_prior[[0, 1, 4]] = True          # edge 4 touches Waste: excluded
        eng = _engine(acc, tmp_path, nodata_anchor_cv=0.3, pin_flat_prior_zeros=True,
                      pin_exclude_sectors=("Waste",))
        cr = eng.run(agg, timesteps=YEARS, persist=False)
        pinned = np.array([0, 1])
        assert (cr.l[pinned] == 0.0).all() and (cr.u[pinned] == 0.0).all()
        rest = np.setdiff1d(np.arange(acc.n_edges), pinned)
        np.testing.assert_array_equal(cr.l[rest], acc.bounds.l[rest])
        np.testing.assert_array_equal(cr.u[rest], acc.bounds.u[rest])

    def test_pin_off_leaves_the_bounds(self, tmp_path):
        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        agg.flat_prior = np.ones(acc.n_edges, dtype=bool)
        cr = _engine(acc, tmp_path, nodata_anchor_cv=0.3).run(agg, timesteps=YEARS,
                                                              persist=False)
        np.testing.assert_array_equal(cr.u, acc.bounds.u)


# ---------------------------------------------------------------------------
# SumConstraintBuilder: σ modes (U3-G4-1, U3-S2-2, U3-S1-1)
# ---------------------------------------------------------------------------

def _spec(**kw):
    spec = {"constraint_id": "sum_x_2020", "variant": "x", "description": "", "edge_ids": [0],
            "coefficients": [1.0], "c_k": 0.0, "sigma_ck_mode": "sd", "sigma_ck_value": 1.0,
            "sigma_ck_fraction": 0.0, "time": "2020", "valuation": "Mass",
            "sector": None, "region": None, "entity": None, "entity_sub": None}
    spec.update(kw)
    return spec


_SUM_COLS = ["constraint_id", "description", "sector_from", "entity_from", "region_from",
             "sector_to", "entity_to", "region_to", "time", "valuation", "coefficient",
             "c_k", "sigma_ck_mode", "sigma_ck_value", "sigma_ck_fraction"]


def _write_sum_csv(tmp_path, **kw):
    row = {"constraint_id": "testsum", "description": "",
           "sector_from": "Mining", "entity_from": "Process", "region_from": "A",
           "sector_to": "Ore", "entity_to": "Flow", "region_to": "A",
           "time": "2020", "valuation": "Mass", "coefficient": "1.0", "c_k": "0",
           "sigma_ck_mode": "sd", "sigma_ck_value": "5.0", "sigma_ck_fraction": ""}
    row.update(kw)
    path = tmp_path / "sum.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUM_COLS)
        w.writeheader()
        w.writerow(row)
    return str(path)


class TestSumSigmaModes:

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="sigma_ck_mode='sdd'"):
            SumConstraintBuilder([_spec(sigma_ck_mode="sdd")])

    def test_unknown_mode_in_a_csv_raises(self, tmp_path):
        path = _write_sum_csv(tmp_path, sigma_ck_mode="sdd")
        with pytest.raises(ValueError, match="sigma_ck_mode='sdd'"):
            SumConstraintBuilder.from_csv(path, _make_toy_acc())

    def test_p_mean_in_a_csv_raises(self, tmp_path):
        """A CSV row has no right-hand-side edges for a p_mean σ_c to propagate over."""
        path = _write_sum_csv(tmp_path, sigma_ck_mode="p_mean")
        with pytest.raises(ValueError, match="p_mean"):
            SumConstraintBuilder.from_csv(path, _make_toy_acc())

    def test_blank_c_k_raises(self, tmp_path):
        path = _write_sum_csv(tmp_path, c_k="")
        with pytest.raises(ValueError, match="blank c_k"):
            SumConstraintBuilder.from_csv(path, _make_toy_acc())

    def test_pedigree_mode_carries_its_precomputed_sigma(self, tmp_path):
        acc = _make_toy_acc()
        path = _write_sum_csv(tmp_path, sigma_ck_mode="pedigree", sigma_ck_value="3.5")
        blk = SumConstraintBuilder.from_csv(path, acc).build_static(acc)
        assert blk.sigma_c[0] == pytest.approx(3.5, rel=1e-12)
        dyn = SumConstraintBuilder.from_csv(path, acc).build_dynamic(
            acc, np.ones(acc.n_edges), np.ones(acc.n_edges), "2020")
        assert dyn.sigma_c[0] == pytest.approx(3.5, rel=1e-12)

    def test_p_mean_propagates_sigma_a_over_the_right_hand_side(self):
        acc = _make_two_valuation_acc()
        b = SumConstraintBuilder.from_acc_struct_subentity(
            acc, entity_filter={"Process"}, rhs_out_entity={"Flow"},
            lhs_neg_out_entity_sub={"Nature"}, sigma_fraction=0.0, p=2.0)
        f = np.arange(1.0, acc.n_edges + 1.0)
        s = np.full(acc.n_edges, 3.0)
        stat, dyn = b.build_static(acc), b.build_dynamic(acc, f, s, "2020")
        rm = dyn.row_meta
        for i, spec in enumerate(b._specs):
            if spec["time"] != "2020":
                assert dyn.sigma_c[i] > 1e9           # deactivated sentinel
                continue
            rhs = spec["_rhs_ids"]
            assert dyn.c[i] == pytest.approx(f[rhs].sum(), rel=1e-12)
            assert dyn.sigma_c[i] == pytest.approx(
                (np.abs(s[rhs]) ** 2.0).sum() ** 0.5, rel=1e-12)
        assert rm["time"].tolist() == stat.row_meta["time"].tolist()

    def test_sigma_a_is_regularised_before_it_propagates_into_p_mean_sigma_c(self, tmp_path):
        """README Stage 3: the σ_a floor flows into a p_mean σ_c (the ordering of the
        engine's step 0b before the per-year overlay)."""
        acc = _make_two_valuation_acc()
        agg = _make_toy_agg_result(acc)
        agg.sigma_agg = np.full(acc.n_edges, 1e-3)
        b = SumConstraintBuilder.from_acc_struct_subentity(
            acc, entity_filter={"Process"}, rhs_out_entity={"Flow"},
            lhs_neg_out_entity_sub={"Nature"}, sigma_fraction=0.0, p=1.3)
        eng = _engine(acc, tmp_path, sigma_a_abs_floor=50.0, p=1.3)
        eng.register(b)
        cr = eng.run(agg, timesteps=["2020"], persist=False)
        active = (cr.row_meta["time"].astype(str) == "2020").to_numpy()
        assert (cr.timestep_data["2020"]["sigma_c"][active] >= 50.0).all()


# ---------------------------------------------------------------------------
# Builders: unknown masks and shapes (U3-G6-2)
# ---------------------------------------------------------------------------

_PARAM_COLS = ["param_id", "param_type", "agg",
               "a_sector_from", "a_entity_from", "a_region_from",
               "a_sector_to", "a_entity_to", "a_region_to", "a_time", "a_valuation",
               "b_sector_from", "b_entity_from", "b_region_from",
               "b_sector_to", "b_entity_to", "b_region_to", "b_time", "b_valuation",
               "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
               "pedigree_reliability", "pedigree_completeness",
               "pedigree_temporal_correlation", "pedigree_geographical_correlation",
               "pedigree_other_correlation"]


def _write_params(tmp_path, **kw):
    """One per-pair param row: numerator (b) Mining/A → Waste/A, denominator (a)
    Mining/A → Ore/A, both 2020; a net-stock row by default."""
    row = {"param_id": "p1", "param_type": "stock_bound", "agg": "",
           "a_sector_from": "Mining", "a_entity_from": "Process", "a_region_from": "A",
           "a_sector_to": "Ore", "a_entity_to": "Flow", "a_region_to": "A",
           "a_time": "2020", "a_valuation": "Mass",
           "b_sector_from": "Mining", "b_entity_from": "Process", "b_region_from": "A",
           "b_sector_to": "Waste", "b_entity_to": "SB", "b_region_to": "A",
           "b_time": "same", "b_valuation": "same",
           "value_nominal": "0", "unc_type": "", "value_sd": "",
           "value_min": "-0.1", "value_max": "0.1"}
    row.update(kw)
    path = tmp_path / "params.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PARAM_COLS)
        w.writeheader()
        w.writerow(row)
    return str(path)


class TestBuilderInputs:

    def test_mass_balance_unknown_mask_raises(self):
        with pytest.raises(ValueError, match="mask not in acc_struct.masks"):
            MassBalanceBuilder(mask="nope").build_static(_make_toy_acc())

    def test_sum_unknown_mask_raises(self):
        acc = _make_toy_acc()
        with pytest.raises(ValueError, match="mask not in acc_struct.masks"):
            SumConstraintBuilder.from_mask_anchor(acc, "nope", np.ones(acc.n_edges),
                                                  sigma_cv=0.1)

    def test_from_mask_anchor_mask_array_of_the_wrong_shape_raises(self):
        acc = _make_toy_acc()
        with pytest.raises(ValueError, match="mask array shape"):
            SumConstraintBuilder.from_mask_anchor(acc, np.ones(3, dtype=bool),
                                                  np.ones(acc.n_edges), sigma_cv=0.1)

    def test_ratio_unknown_mask_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mask not in acc_struct.masks"):
            RatioConstraintBuilder.from_csv(
                _write_params(tmp_path), _make_toy_acc(), mask="nope",
                sigma_rel_floor=0.02, scale_abs_floor=1.0)

    def test_ratio_bound_unknown_mask_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mask not in acc_struct.masks"):
            RatioBoundBuilder.from_csv(_write_params(tmp_path), _make_toy_acc(),
                                       mask="nope")


# ---------------------------------------------------------------------------
# Ratio rows: net stocks and pedigree (U3-S2-4, U3-S5-2, U3-S1-4, U3-G6-3)
# ---------------------------------------------------------------------------

class TestRatioStockRows:

    def _from_csv(self, acc, path, **kw):
        return RatioConstraintBuilder.from_csv(path, acc, sigma_rel_floor=0.02,
                                               scale_abs_floor=1.0, **kw)

    def test_stock_bound_without_value_max_raises(self, tmp_path):
        path = _write_params(tmp_path, value_max="", value_min="")
        with pytest.raises(ValueError, match="no value_max"):
            self._from_csv(_make_toy_acc(), path)

    def test_stock_bound_across_valuations_raises(self, tmp_path):
        acc = _make_two_valuation_acc()
        path = _write_params(tmp_path, b_valuation="CC")
        with pytest.raises(ValueError, match="CC numerator with a Mass denominator"):
            self._from_csv(acc, path)

    def test_stock_bound_within_a_valuation_builds(self, tmp_path):
        acc = _make_two_valuation_acc()
        blk = self._from_csv(acc, _write_params(tmp_path)).build_static(acc)
        assert blk.G_rows.shape[0] == 1

    def test_per_pair_stock_row_is_skipped_on_a_reported_stock_leg(self, tmp_path):
        """#98 on the per-pair path: the row of the year whose stock leg (edge 4, the
        numerator) a feed reports goes; the other year keeps its row."""
        acc = _make_toy_acc()
        path = _write_params(tmp_path, a_time="*")
        assert self._from_csv(acc, path).build_static(acc).G_rows.shape[0] == 2
        covered = np.zeros(acc.n_edges, dtype=bool)
        covered[4] = True
        part = self._from_csv(acc, path, covered=covered).build_static(acc)
        assert part.row_meta["time"].astype(str).tolist() == ["2021"]

    def test_soft_ratio_pedigree_sigma_is_the_total_cv_of_the_nominal(self, tmp_path):
        from pysuto.unc import PedigreeScores
        scores = (2, 3, 1, 2, 4)
        path = _write_params(
            tmp_path, param_type="concentration", value_nominal="0.5", value_min="",
            value_max="", unc_type="pedigree", pedigree_reliability=scores[0],
            pedigree_completeness=scores[1], pedigree_temporal_correlation=scores[2],
            pedigree_geographical_correlation=scores[3], pedigree_other_correlation=scores[4])
        acc = _make_toy_acc()
        blk = self._from_csv(acc, path).build_static(acc)
        # σ_r = σ/|r| with σ = total_cv · nominal, so σ_r = total_cv
        assert blk.sigma_c[0] == pytest.approx(PedigreeScores(*scores).total_cv(),
                                               rel=1e-12)


# ---------------------------------------------------------------------------
# FlowBoundsBuilder (U3-S2-5)
# ---------------------------------------------------------------------------

class TestFlowBoundsLowerBound:

    def test_a_positive_structural_lower_bound_raises(self):
        acc = _make_toy_acc()
        acc.bounds.l[2] = 4.0
        with pytest.raises(ValueError, match="l > 0"):
            FlowBoundsBuilder().build_static(acc)

    def test_disabled_builder_does_not_check(self):
        acc = _make_toy_acc()
        acc.bounds.l[2] = 4.0
        assert FlowBoundsBuilder(enforce_lower=False).build_static(acc).G_rows.shape[0] == 0


# ---------------------------------------------------------------------------
# ConstDiagnostic (U3-G3-1, U3-G5-2, U3-G6-3)
# ---------------------------------------------------------------------------

def _diag_run():
    """Two soft rows, one per year; each year's c holds the other year's row at the
    1e15 sentinel, as a built run does."""
    row_meta = pd.DataFrame([
        {"constraint_id": "sumA", "type": "sum", "variant": "anchor", "sense": "eq",
         "is_hard": False, "time": "1995", "valuation": "Mass"},
        {"constraint_id": "sumB", "type": "sum", "variant": "anchor", "sense": "eq",
         "is_hard": False, "time": "1996", "valuation": "Mass"},
        {"constraint_id": "bnd", "type": "ratio", "variant": "bound", "sense": "le",
         "is_hard": True, "time": "1996", "valuation": "Mass"},
    ])
    G = sp.csr_matrix(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    return types.SimpleNamespace(
        G=G, row_meta=row_meta, a0=np.array([10.0, 5.0, 2.0]), sigma_a=np.ones(3),
        run_id="const_test",
        timestep_data={
            "1995": {"c": np.array([12.0, 0.0, 1e15]), "sigma_c": np.array([3.0, 1e10, 0.0])},
            "1996": {"c": np.array([0.0, 6.0, 1e15]), "sigma_c": np.array([1e10, 3.0, 0.0])},
        })


class TestConstDiagnostic:

    def test_summary_reads_only_the_first_years_rows(self):
        from pysuto.diag import ConstDiagnostic
        text = ConstDiagnostic(_diag_run()).summary()
        assert "Max  (all)  :       2.0000" in text         # |12 − 10|, not the 1e15 sentinels
        assert "Mean (soft) :       2.0000" in text

    def test_plot_writes_four_figures_and_prints_nothing(self, tmp_path, capsys):
        import matplotlib
        matplotlib.use("Agg")
        from pysuto.diag import ConstDiagnostic
        paths = ConstDiagnostic(_diag_run()).plot(odir=str(tmp_path))
        assert len(paths) == 4 and all(__import__("os").path.exists(p) for p in paths)
        assert capsys.readouterr().out == ""

    def test_stats_write_prints_nothing(self, tmp_path, capsys):
        from pysuto.diag import ConstStatsDiagnostic
        paths = ConstStatsDiagnostic(_diag_run(), years=("1995",)).write(str(tmp_path))
        assert len(paths) == 1
        assert capsys.readouterr().out == ""


class TestSafeToken:

    def test_nan_and_none_collapse_to_na(self):
        from pysuto.constraints.naming import _safe
        assert _safe(float("nan"), field="x") == "na"
        assert _safe("nan", field="x") == "na"
        assert _safe(None, field="x") == "na"
