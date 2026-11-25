"""test_gitattributes.py
====================

The `.gitattributes` rule keeps hashed bytes equal on every platform.
"""
from __future__ import annotations

from pathlib import Path

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
    if not ATTRS.exists():  # a wheel or sdist install ships no .gitattributes
        return
    rules = _rules()
    assert rules["*"] == ["text=auto", "eol=lf"]


def test_binary_types_never_converted():
    if not ATTRS.exists():
        return
    rules = _rules()
    for ext in ("docx", "ods", "pdf", "png", "ttf"):
        assert rules["*." + ext] == ["binary"]


def test_crlf_files_keep_committed_bytes():
    if not ATTRS.exists():
        return
    rules = _rules()
    for rel in ("scripts/parse/icsg_parse/icsg_2023f.py",):
        assert rules[rel] == ["-text"]
        assert b"\r\n" in (ROOT / rel).read_bytes()
