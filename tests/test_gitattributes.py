"""test_gitattributes.py
====================

The `.gitattributes` rule keeps hashed bytes equal on every platform.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ROOT / ".gitattributes"


def _rules() -> dict:
    out = {}
    for line in ATTRS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pat, *attrs = line.split()
            out[pat] = attrs
    return out


def test_text_is_lf_whatever_autocrlf():
    if not ATTRS.exists():
        pytest.skip("a wheel or sdist install ships no .gitattributes")
    rules = _rules()
    assert rules["*"] == ["text=auto", "eol=lf"]


def test_binary_types_never_converted():
    if not ATTRS.exists():
        pytest.skip("a wheel or sdist install ships no .gitattributes")
    rules = _rules()
    for ext in ("docx", "ods", "pdf", "png", "ttf"):
        assert rules["*." + ext] == ["binary"]

