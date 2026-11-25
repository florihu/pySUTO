"""test_case_resolution.py — ``--case`` resolves a bundled alias, a dotted module or a
file path (#187): the resolver of :mod:`pysuto.case` that the pipeline's parent and its
step children share, the provenance a resolved case records, and the layering it keeps.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pysuto
from pysuto.case import ALIASES, CaseSpec, case_reference, resolve_case
from pysuto.tutorial import toy

REEXPORT = "from pysuto.tutorial.toy import CASE\n"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def module_file(tmp_path, monkeypatch):
    """``zinc_case_mod.py`` in ``tmp_path``, importable as a module."""
    path = tmp_path / "zinc_case_mod.py"
    path.write_text(REEXPORT)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield path
    sys.modules.pop("zinc_case_mod", None)


class TestForms:

    def test_a_bundled_alias_resolves_and_records_alias_and_the_files_sha(self):
        case = resolve_case("toy")
        assert isinstance(case, CaseSpec) and case.id == "toy"
        assert case.case_source == "alias:toy"
        assert case.case_sha256 == _sha(toy.__file__)

    def test_every_alias_names_a_module_declaring_a_case(self):
        for alias in ALIASES:
            assert resolve_case(alias).case_source == f"alias:{alias}"

    def test_a_dotted_module_resolves_and_records_module_and_the_files_sha(
            self, module_file):
        case = resolve_case("zinc_case_mod")
        assert case.id == "toy" and case.case_source == "module:zinc_case_mod"
        assert case.case_sha256 == _sha(module_file)

    def test_a_file_path_resolves_relative_to_the_working_directory(
            self, tmp_path, monkeypatch):
        path = tmp_path / "zinc_case.py"
        path.write_text(REEXPORT)
        monkeypatch.chdir(tmp_path)
        for ref in ("./zinc_case.py", str(path)):
            case = resolve_case(ref)
            assert case.id == "toy" and case.case_source == f"path:{path}"
            assert case.case_sha256 == _sha(path)

    def test_a_case_file_imports_the_modules_next_to_it_as_a_script_does(
            self, tmp_path, monkeypatch):
        (tmp_path / "zinc_sibling_mod.py").write_text("LABEL = 'zinc'\n")
        path = tmp_path / "zinc_sibling_case.py"
        path.write_text("import zinc_sibling_mod\n" + REEXPORT)
        monkeypatch.setattr(sys, "path", list(sys.path))
        try:
            case = resolve_case(str(path))
        finally:
            sys.modules.pop("zinc_sibling_mod", None)
        assert case.id == "toy" and sys.path[0] == str(tmp_path)

    def test_a_fresh_interpreter_resolves_copper_and_reaches_its_stage1_imports(
            self, tmp_path):
        repo = Path(__file__).resolve().parents[1]
        code = ("from pysuto.case import resolve_case\n"
                f"case = resolve_case({str(repo / 'scripts' / 'copper.py')!r})\n"
                "import copper_feed\n"
                "assert case.id == 'copper' and case.stage1 is not None\n"
                "assert case.acc_id == copper_feed.ACC_ID\n")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(repo / "src")})
        assert proc.returncode == 0, proc.stderr

    def test_the_module_path_of_a_pysuto_module_is_a_module_not_an_alias(self):
        case = resolve_case("pysuto.tutorial.toy")
        assert case.case_source == "module:pysuto.tutorial.toy"
        assert case.case_sha256 == resolve_case("toy").case_sha256

    def test_a_relative_path_is_made_absolute_against_the_given_cwd(self, tmp_path):
        assert case_reference("./a/zinc.py", str(tmp_path)) == str(
            tmp_path / "a" / "zinc.py")
        assert case_reference("toy", str(tmp_path)) == "toy"
        assert case_reference("mylab.zinc", str(tmp_path)) == "mylab.zinc"
        absolute = os.path.abspath("/abs/zinc.py")      # drive-qualified on Windows
        assert case_reference(absolute, str(tmp_path)) == absolute


class TestErrors:

    def test_an_unknown_module_names_the_aliases_and_the_module_tried(self):
        with pytest.raises(ValueError) as exc:
            resolve_case("nope_case_mod")
        msg = str(exc.value)
        assert "nope_case_mod" in msg and "toy" in msg

    def test_an_unknown_path_names_the_path_tried(self, tmp_path):
        ref = str(tmp_path / "missing.py")
        with pytest.raises(ValueError, match="no such file") as exc:
            resolve_case(ref)
        assert ref in str(exc.value)

    def test_a_module_without_a_case_is_refused(self, tmp_path):
        path = tmp_path / "empty_case.py"
        path.write_text("X = 1\n")
        with pytest.raises(ValueError, match="declares no CASE"):
            resolve_case(str(path))

    def test_a_missing_dependency_of_the_module_is_not_reported_as_an_unknown_case(
            self, tmp_path):
        path = tmp_path / "needs_dep.py"
        path.write_text("import no_such_dependency_xyz\n")
        with pytest.raises(ModuleNotFoundError, match="no_such_dependency_xyz"):
            resolve_case(str(path))


class TestProvenance:

    def test_with_knobs_and_the_config_acc_id_keep_the_source(self):
        case = resolve_case("toy")
        knobs = case.imp.knobs()
        name = next(iter(knobs))
        kept = case.with_knobs("imp", {name: knobs[name]})
        assert kept.case_source == "alias:toy" and kept.case_sha256 == case.case_sha256
        assert dataclasses.replace(case, acc_id="x").ref()["case_source"] == "alias:toy"

    def test_provenance_is_left_out_of_equality(self):
        assert resolve_case("toy") == toy.CASE

    def test_a_case_built_in_python_records_neither_field(self):
        assert toy.CASE.ref() == {"id": "toy", "acc_id": toy.CASE.acc_id}
        built = dataclasses.replace(toy.CASE, const=dataclasses.replace(
            toy.CASE.const, builders=lambda ctx: []))
        assert "case_source" not in built.ref() and "case_sha256" not in built.ref()

    def test_ref_and_describe_carry_the_provenance_of_a_resolved_case(self):
        case = resolve_case("toy")
        assert case.ref() == {"id": "toy", "acc_id": toy.CASE.acc_id,
                              "case_source": "alias:toy",
                              "case_sha256": case.case_sha256}
        described = case.describe("imp")
        assert described["case_source"] == "alias:toy" and "imp" in described


class TestSeams:

    def test_the_pipeline_hands_its_children_the_absolute_path(self, tmp_path):
        from pysuto.pipeline import Chain, Pipeline
        (tmp_path / "zinc_case.py").write_text(REEXPORT)
        pipe = Pipeline(steps=("opt",), base=Chain(feed_run_ids=(), const="const_x"),
                        cwd=str(tmp_path), case="./zinc_case.py", data_dir="d")
        spec = pipe._spec("opt", pipe.base, {}, "l", "log")
        assert spec.case == str(tmp_path / "zinc_case.py")

    def test_python_m_pysuto_run_without_case_exits_2_and_lists_the_aliases(
            self, tmp_path):
        src = str(Path(pysuto.__file__).resolve().parents[1])
        proc = subprocess.run([sys.executable, "-m", "pysuto", "run", "--data-dir", "d"],
                              cwd=tmp_path, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": src})
        assert proc.returncode == 2 and "--case is required" in proc.stderr
        assert all(alias in proc.stderr for alias in ALIASES)

    def test_python_m_pysuto_run_with_an_unknown_case_exits_2_naming_what_was_tried(
            self, tmp_path):
        src = str(Path(pysuto.__file__).resolve().parents[1])
        proc = subprocess.run([sys.executable, "-m", "pysuto", "run", "--case",
                               "./nope.py", "--data-dir", "d"], cwd=tmp_path,
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": src})
        assert proc.returncode == 2 and "nope.py" in proc.stderr
        assert "no such file" in proc.stderr


def test_the_resolver_and_the_pipeline_load_no_case_module_until_an_alias_is_resolved():
    """Runtime layering (#187): the library imports a bundled case only when it resolves
    an alias, so importing the resolver, the pipeline and the stages loads none."""
    src = str(Path(pysuto.__file__).resolve().parents[1])
    code = ("import sys, pysuto.case, pysuto.pipeline, pysuto.stages, pysuto._step\n"
            "assert not [m for m in sys.modules if m.startswith('pysuto.tutorial')], "
            "'early'\n"
            "pysuto.case.resolve_case('toy')\n"
            "assert 'pysuto.tutorial.toy' in sys.modules\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": src})
    assert proc.returncode == 0, proc.stderr


class TestResolveFailures:
    """U8-G6-5 (#279): the two branches the docstring promises."""

    def test_an_import_error_inside_the_case_module_propagates(self, tmp_path, monkeypatch):
        (tmp_path / "needs_missing_case.py").write_text(
            "import a_dependency_that_is_not_installed\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        try:
            with pytest.raises(ModuleNotFoundError, match="a_dependency_that_is_not"):
                resolve_case("needs_missing_case")
        finally:
            sys.modules.pop("needs_missing_case", None)

    def test_a_module_with_no_source_file_is_refused(self, monkeypatch):
        import types
        module = types.ModuleType("sourceless_case_mod")
        module.CASE = toy.CASE
        monkeypatch.setitem(sys.modules, "sourceless_case_mod", module)
        with pytest.raises(ValueError, match="no source file"):
            resolve_case("sourceless_case_mod")
