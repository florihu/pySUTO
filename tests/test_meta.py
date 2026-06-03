"""
test_meta.py
============
DataFeedMetadata Excel parsing — focused on the ``row_penalty`` opt-in flag.
"""
from __future__ import annotations

import pandas as pd

from pysuto.meta import DataFeedMetadata


def _write_meta(tmp_path, rows):
    """rows: list of (Meta_name, Value). Write a minimal sidecar xlsx."""
    df = pd.DataFrame(rows, columns=["Meta_name", "Value"])
    path = tmp_path / "meta.xlsx"
    df.to_excel(path, index=False)
    return str(path)


def test_row_penalty_true(tmp_path):
    path = _write_meta(tmp_path, [("Feed_name", "baci"), ("Row_penalty", "TRUE")])
    meta = DataFeedMetadata.from_excel(path)
    assert meta.row_penalty is True
