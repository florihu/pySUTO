"""test_gitattributes_application.py
================================
The CRLF parser of ``scripts/parse/`` keeps its committed bytes. Split from
``tests/test_gitattributes.py`` (#304).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_gitattributes
from test_gitattributes import ATTRS, ROOT, _rules  # noqa: E402


def test_crlf_files_keep_committed_bytes():
    if not ATTRS.exists():
        pytest.skip("a wheel or sdist install ships no .gitattributes")
    rules = _rules()
    for rel in ("scripts/parse/icsg_parse/icsg_2023f.py",):
        assert rules[rel] == ["-text"]
        assert b"\r\n" in (ROOT / rel).read_bytes()
