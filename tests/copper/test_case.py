"""test_case.py — a case is a CaseSpec (#120, #26 §3).

The knobs of a case section are its UPPER_CASE fields; ``--set`` replaces them through
the type checks of :mod:`pysuto.knobs`; a run records its section with functions by
qualified name. The copper case (``scripts/copper.py``, ``CASE``) is the one instance;
the toy sections below are built in this module.
"""
from __future__ import annotations

import ast
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from pysuto.case import (Cascade, CascadeStep, CaseContext, CaseSpec, ConstSection,
                         FillEntry, ImpSection)
import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)
from copper import CASCADE, CASE, CopperConst, CopperImp, builders, terminal
from pysuto.knobs import parse_overrides
from pysuto.registry import _NpEncoder

SRC = Path(__file__).resolve().parents[2] / "src" / "pysuto"


def _toy_terminal(ctx):
    return []


def _toy_builders(ctx):
    return []


@dataclass(frozen=True)
class _ToyImp(ImpSection):
    P_PROP:      float           = 1.3
    TIME_WINDOW: int             = 3
    AREF:        Optional[dict]  = None
    stock_sectors: tuple         = ("Stock",)


@dataclass(frozen=True)
class _ToyConst(ConstSection):
    ANCHOR_CV: float = 0.01
    MODE:      str   = "soft"


def _toy() -> CaseSpec:
    return CaseSpec(id="toy", acc_id="toy", imp=_ToyImp(cascade=(), terminal=_toy_terminal),
                    const=_ToyConst(builders=_toy_builders))


class TestSection:

    def test_the_knobs_are_the_upper_case_fields_in_declaration_order(self):
        assert _toy().imp.knobs() == {"CASCADE_ORDER": (), "CASCADE_MAX_ITER": 10,
                                      "MIN_FLOW_KT": 0.01, "P_PROP": 1.3, "TIME_WINDOW": 3, "AREF": None}
        assert _toy().const.knobs() == {"ANCHOR_CV": 0.01, "MODE": "soft"}

    def test_with_knobs_replaces_the_fields_and_leaves_the_case_alone(self):
        toy = _toy()
        got = toy.with_knobs("const", {"ANCHOR_CV": 0.02})
        assert got.const.ANCHOR_CV == 0.02 and got.const.MODE == "soft"
        assert toy.const.ANCHOR_CV == 0.01
        assert got.imp is toy.imp and got.const.builders is _toy_builders

    def test_a_section_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _toy().const.ANCHOR_CV = 0.02

    @pytest.mark.parametrize("name, value", [("P_PROP", "x"), ("TIME_WINDOW", 1.5),
                                             ("TIME_WINDOW", True)])
    def test_with_knobs_refuses_a_wrongly_typed_value(self, name, value):
        """The library cannot bypass the protocol's checks (pysuto.knobs)."""
        with pytest.raises(TypeError, match="expected"):
            _toy().with_knobs("imp", {name: value})

    def test_with_knobs_keeps_the_three_allowances(self):
        got = _toy().with_knobs("imp", {"P_PROP": 2, "TIME_WINDOW": None, "AREF": {"a": 1}})
        assert (got.imp.P_PROP, got.imp.TIME_WINDOW, got.imp.AREF) == (2, None, {"a": 1})

    @pytest.mark.parametrize("name", ["cascade", "stock_sectors", "NOT_A_KNOB"])
    def test_with_knobs_refuses_a_name_that_is_not_a_knob(self, name):
        with pytest.raises(KeyError, match="not knobs"):
            _toy().with_knobs("imp", {name: 1})

    def test_describe_gives_values_by_value_and_functions_by_qualified_name(self):
        desc = _toy().describe("imp")
        assert desc == {"id": "toy", "acc_id": "toy", "imp": {
            "cascade": [], "terminal": f"{__name__}._toy_terminal", "pre_impute": None,
            "post_impute": None, "stage_config": None, "diag_year": None,
            "CASCADE_ORDER": (), "CASCADE_MAX_ITER": 10, "MIN_FLOW_KT": 0.01,
            "P_PROP": 1.3, "TIME_WINDOW": 3, "AREF": None,
            "stock_sectors": ("Stock",)}}
        assert set(_toy().describe("const")) == {"id", "acc_id", "const"}


class TestSetOverride:
    """``--set KNOB=VALUE`` keeps pysuto.knobs' checks and now lands on the spec."""

    def test_a_set_override_reaches_the_spec_field(self):
        knobs = CASE.const.knobs()
        ov = parse_overrides(["ANCHOR_CV=0.02", "STOCK_BOUNDS_MODE=hard"], knobs, knobs)
        got = CASE.with_knobs("const", ov)
        assert got.const.ANCHOR_CV == 0.02 and got.const.STOCK_BOUNDS_MODE == "hard"
        assert CASE.const.ANCHOR_CV == 0.01

    def test_a_tuple_override_reaches_the_spec_field(self):
        knobs = CASE.imp.knobs()
        order = ("network", "trade", "conc", "ore_grade")
        ov = parse_overrides([f"CASCADE_ORDER={order!r}"], knobs, knobs)
        assert CASE.with_knobs("imp", ov).imp.CASCADE_ORDER == order

    @pytest.mark.parametrize("pair", ["ANCHOR_CV='x'", "SCRAP_LEVEL_ANCHOR=1",
                                      "NODATA_ANCHOR_EXCLUDE=['CO']"])
    def test_a_wrongly_typed_override_raises(self, pair):
        knobs = CASE.const.knobs()
        with pytest.raises(TypeError, match="expected"):
            parse_overrides([pair], knobs, knobs)

    def test_the_three_allowances_hold(self):
        """An int for a float knob, None for any knob, anything for a None default."""
        knobs = CASE.const.knobs()
        ov = parse_overrides(["ANCHOR_CV=1", "SCRAP_SHARE_MODE=None", "PRIOR_BOX_K=2.5"],
                             knobs, knobs)
        got = CASE.with_knobs("const", ov).const
        assert (got.ANCHOR_CV, got.SCRAP_SHARE_MODE, got.PRIOR_BOX_K) == (1, None, 2.5)


class TestCopperCase:

    def test_the_copper_case_declares_its_acc_and_both_sections(self):
        assert (CASE.id, CASE.acc_id) == ("copper", "copper")
        assert isinstance(CASE.imp, CopperImp) and isinstance(CASE.const, CopperConst)
        assert CASE.imp.cascade is CASCADE and CASE.imp.terminal is terminal
        assert CASE.const.builders is builders

    def test_the_copper_spec_serialises_into_a_sidecar(self):
        for stage in ("imp", "const"):
            desc = json.loads(json.dumps(CASE.describe(stage), cls=_NpEncoder))
            assert desc["id"] == "copper"
            assert set(desc[stage]) == {f.name for f in dataclasses.fields(
                getattr(CASE, stage))}
        assert (CASE.describe("imp")["imp"]["terminal"]
                == "copper.terminal")
        assert (CASE.describe("const")["const"]["nodata_anchor_exempt"]
                == "copper.nodata_anchor_exempt")

    def test_the_tied_knob_has_one_value_in_both_sections(self):
        """pysuto.pipeline TIED: imp.P_PROP and const.P_PROP."""
        assert CASE.imp.P_PROP == CASE.const.P_PROP

    def test_the_cascade_order_is_a_permutation_of_the_cascade_ids(self):
        assert sorted(CASE.imp.CASCADE_ORDER) == sorted(e.id for e in CASE.imp.cascade)

    def test_a_cascade_records_its_order_and_pass_cap(self):
        c = Cascade(imputers=[1, 2], max_iter=10)
        assert (c.imputers, c.max_iter) == ([1, 2], 10)


class _Ctx:
    """The part of a CaseContext an entry reads: ``case``, ``acc``, ``logger`` and the
    params registry behind ``use_params``."""

    class _Params:
        def __init__(self, tmp_path):
            self.tmp_path = tmp_path

        def entry(self, pid):
            return {"id": pid, "version": 1}

        def resolve(self, pid):
            return self.tmp_path / pid


def _imp(entries, order=()):
    return _ToyImp(cascade=tuple(entries), terminal=_toy_terminal, CASCADE_ORDER=order)


def _step(i):
    return CascadeStep(i, lambda ctx, i=i: f"imputer {i}")


class TestImpCascade:
    """#136: the cascade is data — entries in declaration order, run in CASCADE_ORDER."""

    def test_the_entries_run_in_the_declared_order_by_default(self):
        got = _imp([_step("a"), _step("b"), _step("c")]).imputers(None)
        assert [c.imputers for c in got] == [["imputer a", "imputer b", "imputer c"]]
        assert got[0].max_iter == 10

    def test_cascade_order_reorders_the_entries(self):
        sec = _imp([_step("a"), _step("b"), _step("c")], order=("c", "a", "b"))
        assert sec.imputers(None)[0].imputers == ["imputer c", "imputer a", "imputer b"]
        assert [e.id for e in sec.ordered()] == ["c", "a", "b"]

    def test_a_case_without_entries_registers_no_cascade(self):
        assert _imp([]).imputers(None) == []

    @pytest.mark.parametrize("order", [("a", "b", "x"), ("a", "a", "b"), ("a",)])
    def test_an_order_that_is_not_a_permutation_is_refused(self, order):
        sec = _imp([_step("a"), _step("b"), _step("c")], order=order)
        with pytest.raises(ValueError, match="permutation"):
            sec.validate()

    def test_a_repeated_entry_id_is_refused(self):
        with pytest.raises(ValueError, match="permutation"):
            _imp([_step("a"), _step("a")]).validate()

    def test_copper_declares_its_cascade_as_fill_entries(self):
        fills = {e.id: e for e in CASE.imp.cascade if isinstance(e, FillEntry)}
        assert list(fills) == ["trade", "conc", "ore_grade"]
        assert [e.id for e in CASE.imp.cascade] == ["trade", "network", "conc", "ore_grade"]
        trade = fills["trade"]
        assert (trade.params_id, trade.rule, trade.sigma_rule, trade.twin) == (
            "conc", "forward", "trade", True)
        assert trade.scope.__name__ == "trade_scope"
        assert (fills["conc"].rule, fills["conc"].scope) == ("both", None)

    def test_the_sidecar_records_each_entrys_source_rule_and_scope(self):
        entries = CASE.describe("imp")["imp"]["cascade"]
        assert entries[0] == {
            "kind": "fill", "id": "trade",
            "source": {"params_id": "conc", "param_types": ["concentration"]},
            "rule": "forward", "sigma_rule": "trade",
            "scope": "pysuto.relation.sources.trade_scope"}
        assert entries[1] == {"kind": "step", "id": "network",
                              "build": "copper.build_network"}
        assert entries[3]["source"] == {
            "params_id": "ore_grade", "param_types": ["concentration", "efficiency"]}
        json.dumps(entries)


def test_use_params_records_an_entry_once_and_returns_its_csv(tmp_path):
    ctx = CaseContext(case=None, acc=None, inputs=type("I", (), {
        "params": _Ctx._Params(tmp_path)})(), data_dir="", agg_result=None, logger=None)
    assert ctx.use_params("conc") == str(tmp_path / "conc" / "params.csv")
    ctx.use_params("eff")
    ctx.use_params("conc")
    assert ctx.param_refs == [{"id": "conc", "version": 1}, {"id": "eff", "version": 1}]


def test_the_library_never_imports_the_tutorial():
    """README §5: ``pysuto.tutorial`` sits at the top of the import direction, beside
    ``__main__``."""
    top = ("tutorial",)
    offenders = []
    for path in SRC.rglob("*.py"):
        if (top[0] in path.relative_to(SRC).parts
                or path.name in ("__main__.py", "_step.py")):
            continue        # the tutorial, ``python -m pysuto`` (#125), its child (#186)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [])
            if any(n == f"pysuto.{t}" or n.startswith(f"pysuto.{t}.") or
                   (isinstance(node, ast.ImportFrom) and node.level and
                    (n == t or n.startswith(f"{t}."))) for n in names for t in top):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == []
