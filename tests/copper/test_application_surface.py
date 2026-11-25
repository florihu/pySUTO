"""test_application_surface.py
===========================
The public surface the copper application imports from ``pysuto`` is declared (#218, #304).

The application's half of ``tests/test_public_surface.py``: ``scripts/copper.py``, its feed
half ``scripts/copper_feed.py`` and the feed preparation ``scripts/feedprep/`` (#302). Every
name they import from ``pysuto`` must resolve, be named in README's module reference and
be in ``__all__`` of its module, where the module declares one. The library's own test
cannot read these files: its CI job deletes ``scripts/``.
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_public_surface
from test_public_surface import INTERNAL, _declared, module_reference  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
APP_FILES = [SCRIPTS / "copper.py", SCRIPTS / "copper_feed.py"]
APP_FILES += sorted((SCRIPTS / "feedprep").glob("*.py"))

# Names the feed preparation imports from the library that README's module reference
# does not name yet. The names are exempt here, not declared there: the module reference
# moves to ``docs/`` in #305, and its table is not this ticket's to edit.
UNDECLARED = {("pysuto.input_registry", "PARAMS_PARENTS"),
              ("pysuto.relation.param_relation", "conc_for")}


def app_imports() -> dict:
    """``{(module, name): {file name, ...}}`` over ``from pysuto... import name``."""
    found: dict = {}
    for path in APP_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.ImportFrom) and not node.level and node.module
                    and node.module.split(".")[0] == "pysuto"):
                for alias in node.names:
                    found.setdefault((node.module, alias.name), set()).add(path.name)
    return found


def test_the_application_files_exist():
    assert len(APP_FILES) >= 5 and all(p.is_file() for p in APP_FILES)


def test_every_imported_name_resolves():
    for (module, name), files in sorted(app_imports().items()):
        mod = importlib.import_module(module)
        if not hasattr(mod, name):
            importlib.import_module(f"{module}.{name}")  # a submodule: raises if absent


def test_every_imported_name_is_in_the_module_reference():
    rows = module_reference()
    missing = sorted(
        f"{module}.{name} ({', '.join(sorted(files))})"
        for (module, name), files in app_imports().items()
        if name not in INTERNAL.get(module, ()) and (module, name) not in UNDECLARED
        and not _declared(rows, module, name))
    assert not missing, "not in README's module reference:\n  " + "\n  ".join(missing)


def test_the_exempt_names_are_still_imported_and_still_undeclared():
    rows = module_reference()
    used = app_imports()
    for module, name in UNDECLARED:
        assert (module, name) in used, f"{module}.{name} is not imported: drop the exemption"
        assert not _declared(rows, module, name), f"{module}.{name} is declared: drop the exemption"


def test_every_imported_name_is_in_all_where_declared():
    bad = []
    for (module, name), files in sorted(app_imports().items()):
        declared = getattr(importlib.import_module(module), "__all__", None)
        if declared is not None and name not in declared:
            bad.append(f"{module}.{name} ({', '.join(sorted(files))})")
    assert not bad, "missing from __all__:\n  " + "\n  ".join(bad)


def test_the_application_imports_no_private_name_the_library_does_not_declare():
    private = {(m, n) for (m, n) in app_imports()
               if any(part.startswith("_") for part in m.split("."))}
    assert private <= {(m, n) for m, names in INTERNAL.items() for n in names}


def test_feedprep_all_names_only_what_exists():
    import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)
    import feedprep
    for name in feedprep.__all__:
        importlib.import_module(f"feedprep.{name}")
