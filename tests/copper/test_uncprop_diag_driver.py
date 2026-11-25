"""test_uncprop_diag_driver.py
===========================
``scripts/uncprop_diag.py`` renders a run under another data directory. Split from
``tests/test_uncprop_diag.py`` (#304); the persisted chain comes from there.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_uncprop_diag
from test_uncprop_diag import CSVS, STEMS, _persisted_chain  # noqa: E402


def test_the_diag_driver_renders_a_run_under_another_data_dir(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    data = tmp_path / "other_data"
    rid, dirs, _ = _persisted_chain(data / "proc", tmp_path, monkeypatch)
    path = Path(__file__).resolve().parents[2] / "scripts" / "uncprop_diag.py"
    spec = importlib.util.spec_from_file_location("uncprop_diag_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.main(rid, data_dir=str(data))
    out = data / "proc" / "uncprop" / rid / "diag"
    assert sorted(p.name for p in out.glob("*.csv")) == sorted(CSVS)
    assert len(list(out.glob("*.png"))) == len(STEMS)
