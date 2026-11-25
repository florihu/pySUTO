"""test_script_declarations.py — the pure parts of two scripts that no test reached (#279).

``scripts/sweep_diag.py`` declares #108's pre-registered outcomes and criteria; its
``declarations()`` is data and is checked without a sweep. ``tools/refresh_tutorial.py``
rewrites machine paths out of the executed notebooks and fails on any ``/home/`` left
(CLAUDE.md "Environment & commands"); ``normalise`` and ``leaks`` are that rule.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str, monkeypatch, folder: Path = SCRIPTS):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class TestSweepDiagDeclarations:

    @pytest.fixture
    def decl(self, monkeypatch):
        return _load("sweep_diag", monkeypatch).declarations()

    def test_outcome_names_are_unique_and_the_every_category_group_is_declared_once(
            self, decl):
        from pysuto.diag.stats.sweep import EveryCategory
        outcomes, _ = decl
        names = [o.name for o in outcomes if not isinstance(o, EveryCategory)]
        assert len(names) == len(set(names))
        assert sum(isinstance(o, EveryCategory) for o in outcomes) == 1

    def test_each_max_abs_names_outcomes_that_are_declared(self, decl):
        from pysuto.diag.stats.sweep import MaxAbs
        outcomes, _ = decl
        declared = {getattr(o, "name", None) for o in outcomes}
        maxabs = [o for o in outcomes if isinstance(o, MaxAbs)]
        assert [m.name for m in maxabs] == ["d_retention[max stage]", "d_total[max stage]"]
        for m in maxabs:
            assert m.parts and all(p.name in declared for p in m.parts)

    def test_the_criteria_are_a6_and_round_e_c_over_declared_outcomes(self, decl):
        outcomes, criteria = decl
        declared = {o.name for o in outcomes if hasattr(o, "name")}
        assert [c.item for c in criteria] == ["A6", "Round-E (c)"]
        for c in criteria:
            assert all(p.name in declared for p in c.outcomes)

    def test_the_export_selector_picks_cross_region_cc_concentrate_legs(self, monkeypatch):
        import pandas as pd
        select = _load("sweep_diag", monkeypatch)._exports("Chile")
        edges = pd.DataFrame({
            "sector_from": ["Concentrate"] * 4, "sector_to": ["Smelting"] * 4,
            "region_from": ["Chile", "Chile", "Chile", "Peru"],
            "region_to":   ["China", "Chile", "China", "China"],
            "valuation":   ["CC", "CC", "Mass", "CC"]})
        assert select(edges).tolist() == [True, False, False, False]


class TestRefreshTutorialPathRule:

    @pytest.fixture
    def rt(self, monkeypatch):
        for mod in ("nbformat", "nbclient"):          # the tutorial extras, not needed here
            monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
        sys.modules["nbclient"].NotebookClient = object
        return _load("refresh_tutorial", monkeypatch, TOOLS)

    def test_the_temp_folder_the_package_and_the_home_are_rewritten(self, rt, tmp_path):
        tmp = tmp_path / "pysuto-tutorial-x"
        work = tmp / "tut"
        pkg = tmp_path / "site" / "pysuto"
        for p in (work, pkg):
            p.mkdir(parents=True)
        text = (f"{work}/runs/imp_1 {tmp}/runs/opt_2 {tmp}/other.csv "
                f"{pkg}/tutorial/x.py {Path.home()}/data path:{work}/case.py")
        out = rt.normalise(text, work, tmp, pkg)
        assert "runs/imp_1" in out and str(work) not in out
        assert f"{tmp}/" not in out and str(pkg) not in out
        assert "pysuto/tutorial/x.py" in out
        assert "~/data" in out
        assert "path:/path/to/tutorial/case.py" in out

    def test_a_surviving_home_or_temp_path_is_reported(self, rt, tmp_path):
        tmp = tmp_path / "pysuto-tutorial-x"
        assert rt.leaks("clean text", tmp) == []
        assert "/home/" in rt.leaks("see /home/someone/x", tmp)
        assert str(tmp) in rt.leaks(f"left {tmp}/runs", tmp)
