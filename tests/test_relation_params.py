"""
test_relation_params.py
=======================
Unit tests for pysuto.relation.params — the one params reader (#128): typed rows, one
parsing policy, each consumer's declared param_types.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.constraints.ratio import ACCEPTS_BOUND, ACCEPTS_EQ, RatioConstraintBuilder
from pysuto.relation import sources
from pysuto.relation.sources import ParamSource
from pysuto.relation.params import SELECTOR_KEYS, ParamRow, read_params


_COLS = ["param_id", "param_type", "agg", "description", *SELECTOR_KEYS,
         "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
         "pedigree_reliability", "pedigree_completeness",
         "pedigree_temporal_correlation", "pedigree_geographical_correlation",
         "pedigree_other_correlation", "epsilon"]


def _rec(**kw) -> dict:
    rec = dict.fromkeys(_COLS, "")
    rec.update(dict.fromkeys(SELECTOR_KEYS, "*"))
    rec.update(param_id="p", param_type="concentration", value_nominal="0.5",
               unc_type="sd", value_sd="0.05")
    rec.update(kw)
    return rec


def _write(path, recs, sep=",", lines_before=()):
    body = pd.DataFrame(recs, columns=_COLS).to_csv(index=False, sep=sep)
    path.write_text("".join(f"{ln}\n" for ln in lines_before) + body)
    return path


class TestReadParams:

    def test_semicolon_and_comma_files_parse_equal(self, tmp_path):
        recs = [_rec(param_id="a", a_sector_from="Ore", value_nominal="0.25"),
                _rec(param_id="b", unc_type="min_max", value_sd="", value_min="0.1",
                     value_max="0.3", agg="true")]
        comma = read_params(_write(tmp_path / "c.csv", recs, ","), {"concentration"})
        semi  = read_params(_write(tmp_path / "s.csv", recs, ";"), {"concentration"})
        # repr, not ==: a dataclass == treats two distinct NaN objects as unequal, and
        # Python 3.13 no longer hands both files the same one
        assert repr(comma) == repr(semi)
        assert [r.param_id for r in comma] == ["a", "b"]

    def test_row_is_typed(self, tmp_path):
        p = _write(tmp_path / "t.csv", [_rec(
            a_sector_from="Ore", a_region_from="", agg="True", epsilon="0.02",
            description="a row")], ",")
        (row,) = read_params(p, {"concentration"})
        assert isinstance(row, ParamRow)
        assert row.agg is True
        assert row.value_nominal == 0.5 and row.value_sd == 0.05
        assert row.epsilon == 0.02
        assert np.isnan(row.value_min)
        assert row.selectors["a_sector_from"] == "Ore"
        assert row.selectors["a_region_from"] == "*"          # blank cell is the wildcard
        assert row.description == "a row"
        assert row.specificity == 1
        assert row.sigma() == 0.05

    def test_comment_row_is_skipped(self, tmp_path):
        p = _write(tmp_path / "c.csv", [_rec(param_id="a")], ",")
        text = p.read_text().splitlines()
        p.write_text("\n".join([text[0], "# a comment row,concentration", *text[1:]]) + "\n")
        assert [r.param_id for r in read_params(p, {"concentration"})] == ["a"]

    def test_foreign_param_type_is_filtered_by_declaration(self, tmp_path, caplog):
        p = _write(tmp_path / "c.csv", [
            _rec(param_id="conc1"),
            _rec(param_id="eff1", param_type="efficiency"),
            _rec(param_id="strip", param_type="stripping_ratio"),
        ], ",")
        with caplog.at_level("INFO"):
            rows = read_params(p, {"concentration", "efficiency"})
        assert [r.param_id for r in rows] == ["conc1", "eff1"]
        assert "1 filtered" in caplog.text and "stripping_ratio" in caplog.text

    def test_the_pair_source_declares_its_types_and_leaves_the_stripping_ratio(self):
        """The params source's pair reading takes concentration / efficiency; the
        stripping_ratio family is outside its PAIR_ACCEPTS."""
        assert sources.PAIR_ACCEPTS == {"concentration", "efficiency"}
        assert "stripping_ratio" not in sources.PAIR_ACCEPTS
        assert "stripping_ratio" in ACCEPTS_EQ
        assert "stripping_ratio" not in ACCEPTS_BOUND

    @pytest.mark.parametrize("field,value", [
        ("value_nominal", "abc"), ("value_max", "x"), ("epsilon", "1,5.2"),
    ])
    def test_malformed_value_raises(self, tmp_path, field, value):
        p = _write(tmp_path / "c.csv", [_rec(**{field: value})], ";")
        with pytest.raises(ValueError, match=field):
            read_params(p, {"concentration"})

    def test_malformed_row_outside_accepts_does_not_raise(self, tmp_path):
        p = _write(tmp_path / "c.csv", [
            _rec(param_id="ok"), _rec(param_id="bad", param_type="efficiency",
                                      value_nominal="abc")], ",")
        assert [r.param_id for r in read_params(p, {"concentration"})] == ["ok"]

    def test_unknown_unc_type_and_missing_sigma_value_raise(self, tmp_path):
        with pytest.raises(ValueError, match="unc_type"):
            read_params(_write(tmp_path / "a.csv", [_rec(unc_type="triangular")]),
                        {"concentration"})
        with pytest.raises(ValueError, match="value_sd"):
            read_params(_write(tmp_path / "b.csv", [_rec(value_sd="")]),
                        {"concentration"})

    def test_missing_param_type_column_raises(self, tmp_path):
        p = tmp_path / "c.csv"
        p.write_text("param_id,value_nominal\nx,1\n")
        with pytest.raises(ValueError, match="param_type"):
            read_params(p, {"concentration"})


class TestSigma:

    def test_sd(self):
        assert ParamRow(unc_type="sd", value_sd=0.5, value_nominal=1.0).sigma() == 0.5

    def test_min_max(self):
        row = ParamRow(unc_type="min_max", value_min=0.2, value_max=0.6, value_nominal=0.4)
        assert row.sigma() == pytest.approx(0.1)

    def test_pedigree_scales_with_nominal(self, tmp_path):
        recs = [_rec(unc_type="pedigree", value_sd="", pedigree_reliability="2",
                     pedigree_completeness="2", pedigree_temporal_correlation="1",
                     pedigree_geographical_correlation="1",
                     pedigree_other_correlation="2", value_nominal=v)
                for v in ("1.0", "2.0")]
        a, b = read_params(_write(tmp_path / "p.csv", recs), {"concentration"})
        assert a.sigma() > 0
        assert b.sigma() == pytest.approx(2 * a.sigma())

    def test_row_without_sigma_rule_raises(self):
        with pytest.raises(ValueError, match="no σ rule"):
            ParamRow(param_id="x", param_type="scrap_share").sigma()


class _Acc:
    """Two nodes, a Mass and a CC edge between them: just what the ratio builders read."""

    def __init__(self) -> None:
        self.node_index = pd.DataFrame(
            {"sector": ["Ore", "Concentrate"], "entity": ["Flow", "Flow"],
             "region": ["A", "A"]}, index=[0, 1])
        self.edge_index = pd.DataFrame({
            "edge_id": [0, 1], "from_id": [0, 0], "to_id": [1, 1],
            "time": ["2020", "2020"], "valuation": ["Mass", "CC"]})
        self.n_edges = 2
        self.masks = {}


class TestSoftRowWithoutSigmaRule:
    """A soft ratio row with no ``unc_type`` raises rather than getting σ = 0 (#128)."""

    def _csv(self, tmp_path, **kw):
        return _write(tmp_path / "scrap.csv", [_rec(
            param_id="scrapshare", param_type="scrap_share", unc_type="", value_sd="",
            value_min="0.0", value_max="0.5", a_valuation="Mass",
            b_valuation="CC", **kw)], ";")

    def test_soft_builder_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no σ rule"):
            RatioConstraintBuilder.from_csv(str(self._csv(tmp_path)), _Acc(), sigma_rel_floor=0.02, scale_abs_floor=1.0)

    def test_hard_builder_does_not_need_one(self, tmp_path):
        b = RatioConstraintBuilder.from_csv(str(self._csv(tmp_path)), _Acc(), hard=True, sigma_rel_floor=0.02, scale_abs_floor=1.0)
        assert len(b._pairs) == 1

    def test_stock_bound_row_reads_its_band_not_a_sigma_rule(self, tmp_path):
        p = _write(tmp_path / "sb.csv", [_rec(
            param_id="netstockbound", param_type="stock_bound", unc_type="", value_sd="",
            value_nominal="0", value_max="0.1", a_valuation="Mass", b_valuation="CC")], ",")
        b = RatioConstraintBuilder.from_csv(str(p), _Acc(), sigma_rel_floor=0.02, scale_abs_floor=1.0)
        assert b._pairs[0].sigma == pytest.approx(0.1 / 1.645)

    def test_the_params_source_raises_on_a_row_without_one(self, tmp_path):
        p = _write(tmp_path / "c.csv", [_rec(unc_type="", value_sd="")], ",")
        with pytest.raises(ValueError, match="no σ rule"):
            ParamSource.from_params(str(p), _Acc())
