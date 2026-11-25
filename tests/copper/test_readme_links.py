"""test_readme_links.py
=====================
Every relative link of README.md points at a file or directory of the repository (#191).
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


def test_every_relative_link_of_the_readme_resolves():
    readme = REPO / "README.md"
    if not readme.is_file():
        pytest.skip("README.md is not part of this tree")
    targets = _relative_targets(readme.read_text(encoding="utf-8"))
    assert targets
    assert [t for t in targets if not (REPO / t).exists()] == []


def test_the_link_finder_ignores_urls_and_anchors():
    text = "[a](docs/x.md#s) [b](https://x.org/y) [c](#top) [d](mailto:z@z.org)"
    assert _relative_targets(text) == ["docs/x.md"]
