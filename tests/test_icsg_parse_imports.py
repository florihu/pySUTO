"""
test_icsg_parse_imports.py
==========================
The ICSG yearbook parsers import the helper module that exists today.

``pysuto.util`` became ``pysuto.utils`` (9aa18b9); three parsers kept the old
name and could not be imported, and no test loaded them (#50).  This one does,
so the next rename fails the suite instead of the next parse run.  Their
``__main__`` blocks are guarded, so importing them reads no data.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))


@pytest.mark.parametrize("module, loader_name", [
    ("icsg_2005",  "icsg_2005"),
    ("icsg_2023f", "icsg_loader"),
    ("icsg_cap",   "extract_icsg_global_series"),
])
def test_parser_imports_and_exposes_its_loader(module, loader_name):
    mod = importlib.import_module(module)
    assert callable(getattr(mod, loader_name))
