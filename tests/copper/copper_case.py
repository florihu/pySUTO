"""copper_case.py — the tests' way to the copper case, which lives in ``scripts/`` (#189).

Importing this module puts ``scripts/`` on ``sys.path`` and re-exports ``copper`` and
``copper_feed``; a test module does ``from copper_case import copper, copper_feed`` (or
``import copper_case`` before ``from copper import …``). It is not a conftest and holds
no fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import copper       # noqa: E402
import copper_feed  # noqa: E402

__all__ = ["copper", "copper_feed"]
