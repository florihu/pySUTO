"""
test_relation_prior.py
======================
The solve-for-prior adapter (:func:`pysuto.relation.prior.solve_lone_unknown`) and the
relations the NetworkImputer shares with Stage 3 (#134): a Flow balance, a CC→Mass twin
and a recipe's pooled efficiency are one object, emitted into ``G`` and solved for a
lone unknown, and an ``agg=true`` eff row that pools another input set than its recipe
raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.constraints.emit import emit
from pysuto.constraints.ratio import RatioConstraintBuilder
from pysuto.impute import NetworkImputer
from pysuto.relation.balance import node_balance_relations
from pysuto.relation.prior import solve_lone_unknown
from pysuto.relation.ratio_relation import conc_twin_relation
from pysuto.relation.relation import Relation

from test_network_imputer import (
    _COLS, MockAcc, _row, _run_logged, write_conc_csv, write_eff_csv,
    write_ore_grade_csv,
)


def _rel(edges, coefficients, rhs=0.0, sense="eq") -> Relation:
    return Relation(
        edges=np.array(edges, dtype=np.int64), coefficients=np.array(coefficients),
        type="massbal", variant="massbal", specs=("X",), time="2020",
        valuation="CC", rhs=rhs, sense=sense,
    )


class TestSolveLoneUnknown:

    def test_solves_the_lone_nan_edge(self):
        f = np.array([np.nan, 3.0, 5.0])
        fill = solve_lone_unknown(_rel([0, 1, 2], [1.0, 1.0, -1.0]), f)
        assert fill.edge == 0 and fill.value == 2.0
        np.testing.assert_array_equal(fill.sources, [1, 2])
        np.testing.assert_array_equal(fill.coefficients, [-1.0, 1.0])

    def test_none_unless_exactly_one_unknown(self):
        rel = _rel([0, 1, 2], [1.0, 1.0, -1.0])
        assert solve_lone_unknown(rel, np.array([np.nan, np.nan, 5.0])) is None
        assert solve_lone_unknown(rel, np.array([1.0, 3.0, 5.0])) is None

    def test_a_nan_outside_scope_counts_as_zero(self):
        rel = _rel([0, 1, 2], [1.0, 1.0, -1.0])
        f = np.array([np.nan, np.nan, 5.0])
        scope = np.array([True, False, True])
        fill = solve_lone_unknown(rel, f, scope)
        assert fill.edge == 0 and fill.value == 5.0
        np.testing.assert_array_equal(fill.sources, [2])     # the NaN is no source

    def test_rhs_enters_the_value(self):
        fill = solve_lone_unknown(_rel([0, 1], [2.0, 1.0], rhs=7.0),
                                  np.array([np.nan, 3.0]))
        assert fill.value == 2.0                             # 2·a + 3 = 7

    def test_a_balance_is_never_a_negative_zero(self):
        f = np.array([np.nan, 4.0, 4.0])
        fill = solve_lone_unknown(_rel([0, 1, 2], [1.0, 1.0, -1.0]), f)
        assert fill.value == 0.0 and not np.signbit(fill.value)

    def test_an_inequality_has_no_value_to_solve_for(self):
        with pytest.raises(ValueError, match="not an equality"):
            solve_lone_unknown(_rel([0, 1], [1.0, -1.0], sense="le"),
                               np.array([np.nan, 1.0]))

    def test_balance_value_is_the_old_expression_bit_for_bit(self):
        rng = np.random.default_rng(134)
        for _ in range(200):
            n_in, n_out = rng.integers(1, 5, size=2)
            vals = rng.lognormal(0.0, 3.0, size=n_in + n_out)
            edges = list(range(n_in + n_out))
            rel = _rel(edges, [1.0] * n_in + [-1.0] * n_out)
            for k in edges:
                f = vals.copy()
                f[k] = np.nan
                inc = edges[:n_in]
                out = edges[n_in:]
                in_sum = float(sum(f[x] for x in inc if x != k and not np.isnan(f[x])))
                out_sum = float(sum(f[x] for x in out if x != k and not np.isnan(f[x])))
                want = (in_sum - out_sum) if k in out else (out_sum - in_sum)
                assert solve_lone_unknown(rel, f).value == want

    def test_twin_value_and_coefficient_are_the_old_expression_bit_for_bit(self):
        rng = np.random.default_rng(1)
        for conc in rng.uniform(0.001, 0.999, size=100):
            rel = conc_twin_relation(0, 1, float(conc), time="2020", valuation="CC")
            f = np.array([rng.lognormal(0.0, 3.0), np.nan])
            fill = solve_lone_unknown(rel, f)
            assert fill.value == float(f[0] / conc)
            assert fill.coefficients.tolist() == [1.0 / float(conc)]


class TestOneObject:
    """§2: the prior adapter's fill coefficients equal the emitted G row."""

    @staticmethod
    def _assert_coefficients_are_the_g_row(rel, edge, n_edges):
        f = np.arange(1.0, n_edges + 1.0)
        f[edge] = np.nan
        fill = solve_lone_unknown(rel, f)
        assert fill.edge == edge
        G = emit([rel], n_edges).G_rows.toarray()[0]
        np.testing.assert_array_equal(fill.coefficients,
                                      -G[fill.sources] / G[fill.edge])
        assert fill.edge_coefficient == G[fill.edge]
        # the filled value satisfies the emitted row
        f[edge] = fill.value
        assert G @ f == pytest.approx(0.0, abs=1e-9)

    def test_every_flow_balance(self):
        acc = MockAcc(times=("2020", "2021"))
        rels = node_balance_relations(acc, "CC", entities={"Flow"})
        assert len(rels) == 6     # Ore, Concentrate, CSXEW × 2 years
        for rel in rels:
            for edge in rel.edges:
                self._assert_coefficients_are_the_g_row(rel, int(edge), acc.n_edges)

    def test_a_conc_twin(self):
        rel = conc_twin_relation(3, 4, 0.275, 0.0375, time="2020", valuation="CC")
        self._assert_coefficients_are_the_g_row(rel, 4, 6)
        np.testing.assert_allclose(rel.sigma, 0.0375 / 0.275, rtol=1e-12)

    def test_the_stage_3_conc_row_is_the_twin_the_network_logs(self, tmp_path):
        # Stage 3's RatioConstraintBuilder row for the Matte conc pair and the network's
        # logged Mass-twin coefficient are the same row: 1/conc = −G_cc / G_mass.
        acc = MockAcc()
        conc = tmp_path / "conc.csv"
        write_conc_csv(conc)
        df = pd.read_csv(conc, sep=";", dtype=str)
        df["param_id"] = df["param_id"].str.replace("_", "")   # ids are underscore-free
        df = df[df["value_nominal"].astype(float) > 0]         # no copperless-WR row
        df.to_csv(conc, sep=";", index=False)
        block = RatioConstraintBuilder.from_csv(str(conc), acc, sigma_rel_floor=0.02, scale_abs_floor=1.0).build_static(acc)
        cc, mass = acc.eid("Smelting", "Matte", "CC"), acc.eid("Smelting", "Matte", "Mass")
        G = block.G_rows.tocsr()
        rows = [i for i in range(G.shape[0]) if G[i, cc] != 0 and G[i, mass] != 0]
        assert len(rows) >= 1
        imp = _network(tmp_path, acc)
        _, _, entries = _run_logged(imp, acc, {("Concentrate", "Smelting", "CC"): 100.0})
        logged = [e for e in entries
                  if e["edge_id"] == mass and e["method"].startswith("network_mass_twin")]
        assert len(logged) == 1
        for i in rows:
            assert logged[0]["coefficients"] == [-G[i, cc] / G[i, mass]]
            assert logged[0]["source_edges"] == [cc]


def _network(tmp_path, acc, eff_rows=None):
    eff, conc, og = (tmp_path / n for n in ("eff.csv", "conc.csv", "og.csv"))
    write_eff_csv(eff)
    if eff_rows is not None:
        pd.DataFrame(eff_rows, columns=_COLS + ["agg"]).to_csv(eff, sep=";", index=False)
    write_conc_csv(conc)
    write_ore_grade_csv(og)
    imp = NetworkImputer(str(eff), str(conc), str(og), p=1.3, cv_cap=float("inf"), q=3.0, alpha_min=0.05)
    imp.prepare(acc)
    return imp


def _agg_smelting_row(a_sector_from="*"):
    """The pooled Smelting efficiency row (A = inputs into Smelting, B = Matte)."""
    return _row(
        param_id="effsmelting", param_type="efficiency", agg="true",
        a_sector_from=a_sector_from, a_entity_from="*", a_region_from="*",
        a_sector_to="Smelting", a_entity_to="Process", a_region_to="same",
        a_time="*", a_valuation="CC",
        b_sector_from="Smelting", b_entity_from="Process", b_region_from="same",
        b_sector_to="Matte", b_entity_to="Flow", b_region_to="same",
        b_time="same", b_valuation="same", value_nominal="0.965",
    )


class TestPooledEff:

    def test_an_agg_row_pooling_the_recipes_inputs_is_its_relation(self, tmp_path):
        acc = MockAcc()
        imp = _network(tmp_path, acc, [_agg_smelting_row()])
        rec = next(r for r in imp._recipes if r.eff_param_id == "effsmelting")
        assert rec.relation.agg and rec.eff == 0.965
        # the recipe's relation is the row Stage 3 emits for the same agg eff row
        eff = tmp_path / "eff.csv"
        block = RatioConstraintBuilder.from_csv(str(eff), acc, sigma_rel_floor=0.02, scale_abs_floor=1.0).build_static(acc)
        mine = emit([rec.relation], acc.n_edges).G_rows.toarray()[0]
        theirs = block.G_rows.toarray()
        assert any(np.array_equal(mine, row) for row in theirs)

    def test_an_agg_row_with_another_input_set_raises(self, tmp_path):
        # Scrap feeds Smelting, but this row pools only the Concentrate input
        with pytest.raises(ValueError, match="pools input edges"):
            _network(tmp_path, MockAcc(), [_agg_smelting_row(a_sector_from="Concentrate")])

    def test_a_process_without_an_eff_row_closes_at_one_hard(self, tmp_path):
        imp = _network(tmp_path, MockAcc())
        cp = next(r for r in imp._recipes if r.eff_param_id is None)
        assert cp.eff == 1.0 and cp.relation.hard and not cp.relation.agg
