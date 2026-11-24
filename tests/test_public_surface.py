"""test_public_surface.py
=====================
The public surface a second case uses is declared (#218).

A case is ``scripts/copper.py`` (with its feed half ``scripts/copper_feed.py``) and the
toy (``pysuto.tutorial.toy*``); the tutorial's notebooks, chapter 10 among them, build an
own case. Every name any of them imports from ``pysuto`` must

* resolve: ``import_module(module)`` has the attribute or the submodule;
* be named in README's module reference (the *Public API* column of the row of its
  module, ``submodule.name`` for a name of a file inside a package row);
* be in ``__all__`` of its module, where the module declares one.

The names of ``pysuto._conditioning`` are the one private exception: the tutorial reaches
for them on purpose (README *Public API*), and the test pins that list.
"""
from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pysuto"
README = ROOT / "README.md"

CASE_FILES = [ROOT / "scripts" / "copper.py", ROOT / "scripts" / "copper_feed.py"]
CASE_FILES += sorted((SRC / "tutorial").glob("*.py"))
NOTEBOOKS = sorted((SRC / "tutorial").glob("*.ipynb"))

# The tutorial's declared reach into the library's private σ conditioning (README
# "Public API"): the toy and chapters 4-5 rebuild the solver's conditioned QP to show it.
INTERNAL = {"pysuto._conditioning": {"condition_prior", "condition_sigma_c"}}


def _sources():
    for path in CASE_FILES:
        yield path, path.read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        # drop IPython magics and shell escapes
        code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith(("%", "!")))
        yield path, code


def case_imports() -> dict:
    """``{(module, name): {file name, ...}}`` over ``from pysuto... import name``."""
    found: dict = {}
    for path, text in _sources():
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.ImportFrom) and not node.level and node.module
                    and node.module.split(".")[0] == "pysuto"):
                for alias in node.names:
                    found.setdefault((node.module, alias.name), set()).add(path.name)
    return found


def module_reference() -> dict:
    """README's module reference: ``{row key: {backticked tokens of its API cell}}``."""
    text = README.read_text(encoding="utf-8")
    body = text[text.index("## Module reference"):text.index("## Workflow")]
    rows: dict = {}
    for line in body.splitlines():
        cells = re.split(r"(?<!\\)\|", line)
        if len(cells) < 4 or "](src/pysuto/" not in cells[1]:
            continue
        tokens = set()
        for t in re.findall(r"`([^`]+)`", cells[2]):
            m = re.match(r"[\w.]+", t.replace("\\", ""))
            if m:
                tokens.add(m.group(0))
        for target in re.findall(r"\]\(src/pysuto/([^)]+)\)", cells[1]):
            key = target.rstrip("/")
            key = key[:-3] if key.endswith(".py") else key
            rows.setdefault(key.replace("/", "."), set()).update(tokens)
    return rows


def _declared(rows: dict, module: str, name: str) -> bool:
    """Is ``name`` of ``pysuto.<module>`` named in the row of the module or its package?"""
    parts = module.split(".")[1:]
    if not parts:                       # ``from pysuto import <submodule>``: its own row
        return name in rows
    for cut in range(len(parts), 0, -1):
        tokens = rows.get(".".join(parts[:cut]))
        if tokens is None:
            continue
        rest = parts[cut:]
        return name in tokens or (bool(rest) and f"{rest[-1]}.{name}" in tokens)
    return False


def test_every_imported_name_resolves():
    for (module, name), files in sorted(case_imports().items()):
        mod = importlib.import_module(module)
        if not hasattr(mod, name):
            importlib.import_module(f"{module}.{name}")  # a submodule: raises if absent


def test_every_imported_name_is_in_the_module_reference():
    rows = module_reference()
    missing = sorted(
        f"{module}.{name} ({', '.join(sorted(files))})"
        for (module, name), files in case_imports().items()
        if name not in INTERNAL.get(module, ()) and not _declared(rows, module, name))
    assert not missing, "not in README's module reference:\n  " + "\n  ".join(missing)


def test_every_imported_name_is_in_all_where_declared():
    bad = []
    for (module, name), files in sorted(case_imports().items()):
        declared = getattr(importlib.import_module(module), "__all__", None)
        if declared is not None and name not in declared:
            bad.append(f"{module}.{name} ({', '.join(sorted(files))})")
    assert not bad, "missing from __all__:\n  " + "\n  ".join(bad)


def test_the_only_private_imports_are_the_declared_ones():
    private = {
        (module, name) for (module, name) in case_imports()
        if any(part.startswith("_") for part in module.split("."))}
    declared = {(m, n) for m, names in INTERNAL.items() for n in names}
    assert private == declared
    text = README.read_text(encoding="utf-8")
    assert "pysuto._conditioning" in text[text.index("### Public API"):].split("\n## ")[0]
