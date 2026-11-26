"""test_import_direction.py
=======================
The import direction of docs/architecture.md (*Import direction*) is checked, not only drawn (#217).

The layer table of docs/architecture.md (``| unit | module-level imports | typing only |``) is the
one source: this module reads it and walks ``src/pysuto`` with ``ast``. A *unit* is a
top-level module or subpackage of ``pysuto``; edges inside a unit are free. Three rules:

* a module-level import (including one in a ``try`` at module level) may point only at
  a unit its own row lists, an import under ``if TYPE_CHECKING:`` only at a unit its
  row lists as typing only: a back-edge is refused;
* no unit but ``__main__`` imports ``tutorial``, and the units of the table are exactly
  the units of the tree;
* an import of ``pysuto`` inside a function says why, in a comment on its line or above
  it (a run of imports shares the comment above the first): ``cycle``, ``lazy`` or
  ``optional``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pysuto"
README = ROOT / "docs" / "architecture.md"
REASONS = ("cycle", "lazy", "optional")


def _unit(module: str) -> str | None:
    """``pysuto.acc.x`` -> ``acc``; the bare package ``pysuto`` -> ``None``."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def read_table() -> dict:
    """The architecture document's layer table: ``{unit: (module-level set, typing-only set)}``."""
    text = README.read_text(encoding="utf-8")
    start = text.index("**Import direction is strictly one-way")
    rows = {}
    for line in text[start:].splitlines():
        if line.startswith("## "):
            break
        m = re.match(r"\|\s*`(\w+)`\s*\|([^|]*)\|([^|]*)\|\s*$", line)
        if m:
            names = lambda cell: set(re.findall(r"`(\w+|\*)`", cell))
            rows[m.group(1)] = (names(m.group(2)), names(m.group(3)))
    return rows


def _targets(node, module: str, is_pkg: bool) -> list:
    """The ``pysuto`` units an import statement points at."""
    if isinstance(node, ast.Import):
        return [_unit(a.name) for a in node.names if a.name.split(".")[0] == "pysuto"]
    if node.level:
        base = module.split(".")
        base = base if is_pkg else base[:-1]
        base = base[: len(base) - (node.level - 1)]
        full = ".".join(base + ([node.module] if node.module else []))
    else:
        full = node.module or ""
    if full.split(".")[0] != "pysuto":
        return []
    if full == "pysuto":                       # ``from pysuto import stages``
        return [a.name for a in node.names]
    return [_unit(full)]


def imports(path: Path):
    """Yield ``(unit, kind, line, target, node, lines)`` for every pysuto import.

    ``kind`` is ``module``, ``typing`` (under ``if TYPE_CHECKING``) or ``local``
    (inside a function or method).
    """
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    module = ".".join(["pysuto"] + parts)
    unit = parts[0] if parts else None
    lines = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))

    def walk(node, kind):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, "local")
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for t in _targets(child, module, is_pkg):
                    if t is not None:
                        yield unit, kind, child.lineno, t, child, lines
            elif (isinstance(child, ast.If) and kind == "module"
                  and "TYPE_CHECKING" in ast.dump(child.test)):
                yield from walk(child, "typing")
            else:
                yield from walk(child, kind)

    yield from walk(tree, "module")


def all_imports():
    for path in sorted(SRC.rglob("*.py")):
        yield from imports(path)


def tree_units() -> set:
    units = set()
    for path in SRC.glob("*"):
        if path.name == "__pycache__":
            continue
        if path.suffix == ".py":
            units.add(path.stem)
        elif path.is_dir() and (path / "__init__.py").exists():
            units.add(path.name)
    return units - {"__init__"}


def check_edges(edges, table: dict) -> list:
    """The violations among ``edges`` (``imports`` tuples) against ``table``."""
    bad = []
    for unit, kind, line, target, _node, _lines in edges:
        if unit is None or unit == target or kind == "local":
            continue
        module_ok, typing_ok = table.get(unit, (set(), set()))
        if target == "tutorial" and unit != "__main__":
            bad.append(f"{unit}:{line} imports tutorial")
        elif "*" in module_ok:
            continue
        elif kind == "module" and target not in module_ok:
            bad.append(f"{unit}:{line} -> {target} (module level)")
        elif kind == "typing" and target not in module_ok | typing_ok:
            bad.append(f"{unit}:{line} -> {target} (typing only)")
    return bad


def _has_reason(lines: list, lineno: int) -> bool:
    """A reason keyword in the comment of line ``lineno`` or the comment block above
    the run of imports it belongs to."""
    def comment(i):
        return lines[i].partition("#")[2].lower() if "#" in lines[i] else ""

    i = lineno - 1
    if any(r in comment(i) for r in REASONS):
        return True
    while i > 0 and re.match(r"\s*(from|import)\s", lines[i - 1]):
        i -= 1
        if any(r in comment(i) for r in REASONS):
            return True
    j = i - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        if any(r in lines[j].lower() for r in REASONS):
            return True
        j -= 1
    return False


def missing_reasons(edges) -> list:
    return [f"{u}:{line} local import without a reason comment"
            for u, kind, line, _t, _n, lines in edges
            if kind == "local" and not _has_reason(lines, line)]


def test_table_names_exactly_the_units_of_the_tree():
    table = read_table()
    assert set(table) == tree_units()
    for unit, (module_ok, typing_ok) in table.items():
        assert (module_ok | typing_ok) - {"*"} <= set(table), unit


def test_no_module_level_import_runs_against_the_table():
    assert check_edges(all_imports(), read_table()) == []


def test_a_planted_back_edge_is_refused():
    table = read_table()
    planted = [("lp", "module", 1, "solver", None, [])]
    assert check_edges(planted, table) == ["lp:1 -> solver (module level)"]
    assert check_edges([("registry", "typing", 3, "solver", None, [])], table) == []
    assert check_edges([("diag", "module", 2, "tutorial", None, [])], table) != []


def test_every_local_import_says_why():
    assert missing_reasons(all_imports()) == []


def test_a_local_import_without_a_comment_is_refused():
    src = ["def f():", "    from pysuto.acc import X", "    from pysuto.lp import Y"]
    assert missing_reasons([("x", "local", 2, "acc", None, src)]) != []
    src[1] = "    from pysuto.acc import X   # lazy: heavy"
    assert missing_reasons([("x", "local", 3, "lp", None, src)]) == []
    assert missing_reasons([("x", "local", 2, "acc", None, src)]) == []
