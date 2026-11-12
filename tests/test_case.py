"""test_case.py — a case is a CaseSpec (#120, #26 §3).

The knobs of a case section are its UPPER_CASE fields; ``--set`` replaces them through
the type checks of :mod:`pysuto.knobs`; a run records its section with functions by
qualified name. The copper case (``pysuto.cases.copper.CASE``) is the one instance;
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

from pysuto.case import Cascade, CaseSpec, ConstSection, ImpSection
from pysuto.cases.copper import CASE, CopperConst, CopperImp, builders, imputers
from pysuto.knobs import parse_overrides
from pysuto.registry import _NpEncoder

SRC = Path(__file__).resolve().parents[1] / "src" / "pysuto"


def _toy_imputers(ctx):
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
    return CaseSpec(id="toy", acc_id="toy", imp=_ToyImp(imputers=_toy_imputers),
                    const=_ToyConst(builders=_toy_builders))


class TestSection:

    def test_the_knobs_are_the_upper_case_fields_in_declaration_order(self):
        assert _toy().imp.knobs() == {"P_PROP": 1.3, "TIME_WINDOW": 3, "AREF": None}
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

    @pytest.mark.parametrize("name", ["imputers", "stock_sectors", "NOT_A_KNOB"])
    def test_with_knobs_refuses_a_name_that_is_not_a_knob(self, name):
        with pytest.raises(KeyError, match="not knobs"):
            _toy().with_knobs("imp", {name: 1})

    def test_describe_gives_values_by_value_and_functions_by_qualified_name(self):
        desc = _toy().describe("imp")
        assert desc == {"id": "toy", "acc_id": "toy", "imp": {
            "imputers": f"{__name__}._toy_imputers", "pre_impute": None,
            "post_impute": None, "P_PROP": 1.3, "TIME_WINDOW": 3, "AREF": None,
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
        order = ("network", "trade", "conc", "ore_grade", "eff")
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
        assert CASE.imp.imputers is imputers and CASE.const.builders is builders

    def test_the_copper_spec_serialises_into_a_sidecar(self):
        for stage in ("imp", "const"):
            desc = json.loads(json.dumps(CASE.describe(stage), cls=_NpEncoder))
            assert desc["id"] == "copper"
            assert set(desc[stage]) == {f.name for f in dataclasses.fields(
                getattr(CASE, stage))}
        assert (CASE.describe("imp")["imp"]["imputers"]
                == "pysuto.cases.copper.imputers")
        assert (CASE.describe("const")["const"]["nodata_anchor_exempt"]
                == "pysuto.cases.copper.nodata_anchor_exempt")

    def test_the_tied_knob_has_one_value_in_both_sections(self):
        """pysuto.pipeline TIED: imp.P_PROP and const.P_PROP."""
        assert CASE.imp.P_PROP == CASE.const.P_PROP

    def test_the_cascade_order_is_a_permutation_of_the_cascade_ids(self):
        assert sorted(CASE.imp.CASCADE_ORDER) == sorted(CASE.imp.cascade_ids)

    def test_a_cascade_records_its_order_and_pass_cap(self):
        c = Cascade(imputers={"a": 1}, order=("a",), max_iter=10)
        assert (c.order, c.max_iter) == (("a",), 10)


def test_the_library_never_imports_the_cases():
    """README §5: pysuto.cases sits at the top of the import direction."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if "cases" in path.relative_to(SRC).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [])
            if any(n == "pysuto.cases" or n.startswith("pysuto.cases.") or
                   (isinstance(node, ast.ImportFrom) and node.level and
                    (n == "cases" or n.startswith("cases."))) for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == []
