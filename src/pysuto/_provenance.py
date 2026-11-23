"""
_provenance.py
==============
The two provenance helpers every registry and the pipeline write with: the UTC ISO-8601
timestamp (``2026-11-23T10:15:00Z``, seconds resolution) and the sha256 of a file. A leaf
module: it imports nothing from pysuto, so ``registry``, ``input_registry``, ``pipeline``
and ``case`` share one copy (#279).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


def now_iso() -> str:
    """Current UTC time as ISO-8601 with trailing ``Z`` (seconds resolution)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Union[str, Path]) -> str:
    """Hex sha256 of the file at *path*, read in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
