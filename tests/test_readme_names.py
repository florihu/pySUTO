"""test_readme_names.py
=====================
The code names of README.md match the code (#225): every backticked code name exists in
the repository, every module file of ``src/pysuto/`` and every ``python -m pysuto`` subcommand
and flag is named in README. A rename or a new module that leaves README behind turns
this test red.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts", "tests", "config")
CODE_SUFFIXES = (".py", ".yaml", ".yml", ".toml")
TICK = re.compile(r"`([^`\n]+)`")
RUN_ID = re.compile(r"(feed|imp|const|opt|uncprop|psut|sweep)_\w*\d{8}_\d{6}_[0-9a-f]{8}")
UPPER = re.compile(r"[A-Z][A-Z0-9_]{3,}")
SNAKE = re.compile(r"[a-z_][a-z0-9_]*_[a-z0-9_]*")
CAMEL = re.compile(r"[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*")
DOTTED = re.compile(r"pysuto(\.[A-Za-z_][A-Za-z0-9_]*)+")
PYFILE = re.compile(r"[\w/.-]+\.py")

# Names that README names on purpose and the code lacks.
EXEMPT = {
    "./my_case.py",          # the file name of a user's case in the README walk-through
    "conftest.py",           # README says the repo has none, on purpose (CLAUDE.md, Testing)
    "BoundedKalmanUncProp",  # a future Stage 5 concrete README names beside BaseUncProp
}


def _code_names(root: Path) -> tuple:
    """Return the words of the code and the file names of the repository.

    The words come from ``src``, ``scripts``, ``tests``, ``config`` and ``pyproject.toml``;
    this test file is left out, so its own exemptions do not vouch for themselves.
    """
    files = [p for d in CODE_DIRS for p in (root / d).rglob("*")
             if p.is_file() and p.suffix in CODE_SUFFIXES]
    files.append(root / "pyproject.toml")
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in files
                     if p.is_file() and p.resolve() != Path(__file__).resolve())
    return set(re.findall(r"\w+", text)), {p.name for p in files}


def _stale_names(readme: str, root: Path) -> list:
    """Return the backticked code names of ``readme`` that ``root`` lacks, sorted."""
    words, file_names = _code_names(root)
    stale = []
    for tick in sorted(set(TICK.findall(readme))):
        if tick in EXEMPT or RUN_ID.fullmatch(tick):
            continue
        if UPPER.fullmatch(tick) or SNAKE.fullmatch(tick) or CAMEL.fullmatch(tick):
            if tick not in words:
                stale.append(tick)
        elif DOTTED.fullmatch(tick):
            parts, path = tick.split("."), root / "src"
            used = 0
            for part in parts:
                if (path / part).is_dir():
                    path = path / part
                elif (path / (part + ".py")).is_file():
                    path = path / (part + ".py")
                else:
                    break
                used += 1
                if path.is_file():
                    break
            if used < 2 or not all(p in words for p in parts[used:]):
                stale.append(tick)
        elif PYFILE.fullmatch(tick):
            if tick.rsplit("/", 1)[-1] not in file_names:
                stale.append(tick)
    return stale


def _undocumented_modules(readme: str, root: Path) -> list:
    """Return the modules of ``src/pysuto`` whose file name ``readme`` lacks.

    A dotted path alone does not count: ``pysuto._step`` in prose leaves the file unnamed. The
    file name must stand alone, so ``data.py`` does not name ``a.py``.
    """
    missing = []
    for path in sorted((root / "src" / "pysuto").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        if not re.search(r"(?<![\w-])%s(?!\w)" % re.escape(path.name), readme):
            missing.append(path.relative_to(root).as_posix())
    return missing


def _cli_words(parser: argparse.ArgumentParser) -> list:
    """Return the subcommands and ``--flags`` of an argparse parser, sorted."""
    words = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                words.add(name)
                words.update(_cli_words(sub))
        words.update(o for o in action.option_strings if o.startswith("--"))
    words.discard("--help")
    return sorted(words)


def _readme() -> str:
    readme = REPO / "README.md"
    if not readme.is_file() or not (REPO / "src" / "pysuto").is_dir():
        pytest.skip("README.md and src/ are not part of this tree")
    return readme.read_text(encoding="utf-8")


def test_every_code_name_of_the_readme_exists_in_the_code():
    stale = _stale_names(_readme(), REPO)
    assert stale == [], "README names that the code lacks: " + ", ".join(stale)


def test_every_module_of_the_package_is_named_in_the_readme():
    missing = _undocumented_modules(_readme(), REPO)
    assert missing == [], "modules README does not name: " + ", ".join(missing)


def test_every_subcommand_and_flag_of_the_cli_is_named_in_the_readme():
    from pysuto.__main__ import _parser

    readme = _readme()
    words = _cli_words(_parser())
    assert "run" in words and "--data-dir" in words
    missing = [w for w in words if not re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(w),
                                                 readme)]
    assert missing == [], "CLI words README does not name: " + ", ".join(missing)


def test_the_name_finder_flags_only_missing_code_names(tmp_path):
    (tmp_path / "src" / "pysuto").mkdir(parents=True)
    (tmp_path / "src" / "pysuto" / "real.py").write_text("KEEP_ME = 1\nlive_name = 2\n")
    for d in ("scripts", "tests", "config"):
        (tmp_path / d).mkdir()
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src" / "pysuto" / "a.py").write_text("class LiveClass:\n    pass\n")
    readme = ("`KEEP_ME` `GONE_ONE` `live_name` `dead_name` `pysuto.real` "
              "`pysuto.real.live_name` `pysuto.gone` `real.py` `gone.py` `plain` `ABC` "
              "`LiveClass` `GoneClass` `opt_20261119_130302_b20f0dba` `conftest.py`")
    assert _stale_names(readme, tmp_path) == [
        "GONE_ONE", "GoneClass", "dead_name", "gone.py", "pysuto.gone"]
    assert _undocumented_modules("`real.py` [a.py](src/pysuto/a.py)", tmp_path) == []
    assert _undocumented_modules("`real.py` `data.py`", tmp_path) == ["src/pysuto/a.py"]
