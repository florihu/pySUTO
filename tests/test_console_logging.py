"""The console log handler survives a stream that cannot encode a character.

A pipe on Windows encodes with cp1252, which lacks ``≈``; the sweep driver's dry run
logs one (``plan: ... ≈ ...``) and lost the line.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from pathlib import Path

from pysuto.logging import _ConsoleHandler

SRC = Path(__file__).resolve().parents[1] / "src"


def test_a_character_the_stream_cannot_encode_is_replaced_not_dropped():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")
    handler = _ConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "plan: 3 runs ≈ 5 min", (), None)
    handler.emit(record)
    stream.flush()
    assert raw.getvalue() == b"plan: 3 runs ? 5 min\n"


def test_a_utf8_stream_keeps_the_character():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    handler = _ConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "≈", (), None))
    stream.flush()
    assert raw.getvalue() == "≈\n".encode("utf-8")


def test_the_package_command_writes_its_errors_as_utf8_when_the_locale_is_cp1252(tmp_path):
    """The parent ``python -m pysuto`` writes UTF-8 like the steps it starts: a reader that
    decodes its stderr as UTF-8 gets the path back, "σ" and "–" included."""
    proc = subprocess.run([sys.executable, "-m", "pysuto", "run", "--case", "./missing_σ–case.py",
                           "--data-dir", str(tmp_path / "data")], cwd=str(tmp_path),
                          capture_output=True,
                          env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONPATH": str(SRC)})
    assert proc.returncode == 2
    assert "missing_σ–case.py" in proc.stderr.decode("utf-8")
