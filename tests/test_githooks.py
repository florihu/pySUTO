"""The commit-msg hook refuses GitHub closing keywords.

GitHub closes an issue when a commit on the default branch says "fix #98": a
merge message once closed #98 before its measurement, and a sentence about
"the fixed #63/#95 items" reads the same way to GitHub. Issues close through
``gh issue close --comment`` with their REP; a commit names an issue bare.
"""
from __future__ import annotations

import os
import subprocess

import pytest

HOOK = os.path.join(os.path.dirname(__file__), "..", ".githooks", "commit-msg")


def _run(tmp_path, message):
    f = tmp_path / "COMMIT_EDITMSG"
    f.write_text(message, encoding="utf-8")
    return subprocess.run(["bash", HOOK, str(f)], capture_output=True, text=True)


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
