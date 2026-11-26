"""test_readme_links.py
=====================
Every relative link of the documentation set points at a file or directory of the repository
(#191): README.md, docs/architecture.md, docs/workflow.md, scripts/README.md (#305) and
CONTRIBUTING.md (#339). A commit that touches only one of them runs these tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\]\(([^)\s]+)")


def _relative_targets(readme: str) -> list:
    targets = (m.group(1).split("#")[0] for m in LINK.finditer(readme))
    return sorted({t for t in targets if t and not t.startswith(("http", "mailto:"))})


DOCUMENTS = ("README.md", "docs/architecture.md", "docs/workflow.md", "scripts/README.md",
             "CONTRIBUTING.md")


@pytest.mark.parametrize("name", DOCUMENTS)
def test_every_relative_link_of_the_document_resolves(name):
    document = REPO / name
    if not document.is_file():
        pytest.skip(name + " is not part of this tree")
    targets = _relative_targets(document.read_text(encoding="utf-8"))
    assert targets
    assert [t for t in targets if not (document.parent / t).exists()] == []


def test_the_link_finder_ignores_urls_and_anchors():
    text = "[a](docs/x.md#s) [b](https://x.org/y) [c](#top) [d](mailto:z@z.org)"
    assert _relative_targets(text) == ["docs/x.md"]


def test_the_pre_commit_hook_runs_the_readme_tests_on_a_commit_to_any_document():
    """The hook's README-only trigger names every document these tests check."""
    hook    = (REPO / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    trigger = next(ln for ln in hook.splitlines() if "grep -qxE" in ln).split("'")[1]
    assert [name for name in DOCUMENTS if not re.fullmatch(trigger, name)] == []
