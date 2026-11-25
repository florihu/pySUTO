"""The console log handler survives a stream that cannot encode a character.

A pipe on Windows encodes with cp1252, which lacks ``≈``; the sweep driver's dry run
logs one (``plan: ... ≈ ...``) and lost the line.
"""
from __future__ import annotations

import io
import logging

from pysuto.logging import _ConsoleHandler


def test_a_character_the_stream_cannot_encode_is_replaced_not_dropped():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    handler = _ConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "plan: 3 runs ≈ 5 min", (), None)
    handler.emit(record)
    stream.flush()
    assert raw.getvalue() == b"plan: 3 runs ? 5 min\n"


def test_a_utf8_stream_keeps_the_character():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8")
    handler = _ConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "≈", (), None))
    stream.flush()
    assert raw.getvalue() == "≈\n".encode("utf-8")
