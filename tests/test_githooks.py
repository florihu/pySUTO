"""The commit-msg hook refuses GitHub closing keywords.

GitHub closes an issue when a commit on the default branch says "fix #98": a
merge message once closed #98 before its measurement, and a sentence about
"the fixed #63/#95 items" reads the same way to GitHub. Issues close through
``gh issue close --comment`` with their REP; a commit names an issue bare.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = (Path(__file__).resolve().parents[1] / ".githooks" / "commit-msg").as_posix()
PRE_COMMIT = Path(__file__).resolve().parents[1] / ".githooks" / "pre-commit"


def _bash():
    """Return a bash that runs the hook: on Windows Git's, never WSL's ``bash.exe``."""
    if os.name != "nt":
        return "bash"
    git = shutil.which("git")
    if git:                       # <git root>/cmd/git.exe or <git root>/bin/git.exe
        here = os.path.dirname(os.path.abspath(git))
        for root in (os.path.dirname(here), here):
            for rel in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
                cand = os.path.join(root, *rel)
                if os.path.isfile(cand):
                    return cand
    pytest.skip("Git's bash is not installed; PATH's bash.exe may be WSL's, which cannot "
                "run the hook")


def _run(tmp_path, message):
    f = tmp_path / "COMMIT_EDITMSG"
    f.write_text(message, encoding="utf-8")
    return subprocess.run([_bash(), HOOK, f.as_posix()], capture_output=True, text=True)


@pytest.mark.parametrize("message", [
    "fix(feed): the netting refuses a stale input (#63)",
    "measure(constraints): PSC rows ship (ADR 0040, #98, #36)",
    "Merge branch 'fix/psc-rows': PSC rows skip reported legs (#98)",
    "docs(rep): the prefix #3 and suffixes stay legal",
    "fix(parse): a merged cell drops its Electrowon type (#110)\n\n# fixes #110 in a comment line",
    "test(hooks): the hook reads the message only (#98)\n"
    "# ------------------------ >8 ------------------------\n"
    "+    \"Merge branch 'fix/psc-rows': fix #98\",",
])
def test_a_message_naming_issues_bare_passes(tmp_path, message):
    assert _run(tmp_path, message).returncode == 0


@pytest.mark.parametrize("message", [
    "Merge branch 'fix/psc-rows': PSC rows skip reported legs, fix #98",
    "docs(manuscript): the fixed #63/#95/#98 items leave the limitations",
    "feat(diag): calibration, closes #87",
    "refactor: Resolves: #12",
    "chore: fixes florihu/pySUTO#53",
    "docs: Fixes https://github.com/florihu/pySUTO/issues/98",
])
def test_a_closing_keyword_before_an_issue_is_refused(tmp_path, message):
    r = _run(tmp_path, message)
    assert r.returncode == 1
    assert "gh issue close" in r.stderr


def test_every_test_file_the_pre_commit_hook_names_exists():
    """A commit that touches only the README runs the test files the hook names, so a moved
    test file must be renamed in the hook too."""
    root  = PRE_COMMIT.parents[1]
    lines = [ln.strip() for ln in PRE_COMMIT.read_text(encoding="utf-8").splitlines()]
    named = next(ln for ln in lines if ln.startswith('TESTS="tests/') and ln != 'TESTS="tests/"')
    for rel in named.split('"')[1].split():
        if rel.startswith("tests/copper/") and not (root / "tests" / "copper").is_dir():
            continue                     # the library job runs without the application's tests
        assert (root / rel).is_file(), rel
