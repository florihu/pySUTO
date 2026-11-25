"""test_tool_declarations.py — the pure part of a tool that no test reached (#279).

``tools/refresh_tutorial.py`` rewrites machine paths out of the executed notebooks and
fails on any ``/home/`` left (CLAUDE.md "Environment & commands"); ``normalise`` and
``leaks`` are that rule. Split from ``tests/copper/test_script_declarations.py`` (#304):
the tooling is not the application.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str, monkeypatch, folder: Path = TOOLS):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


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
