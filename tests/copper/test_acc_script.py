"""test_acc_script.py
==================
``scripts/acc.py`` picks the reader from what ``current/`` holds: the acc.json route builds
what the ods route builds, and a folder holding both is refused. Split from
``tests/test_acc_spec.py`` (#304), whose spec readers need no application; the fixtures
come from there.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_acc_spec
from test_acc_spec import _write_json_case, _write_ods_case  # noqa: E402

REPO = Path(__file__).resolve().parents[2]



def _run_acc_script(monkeypatch, tmp_path: Path, input_root: Path, out: Path) -> None:
    script = REPO / "scripts" / "acc.py"
    spec = importlib.util.spec_from_file_location("_acc_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)                       # logs/ lands in the tmp dir
    monkeypatch.setattr(sys, "argv", ["acc.py", "--id", "toy",
                                      "--input-root", str(input_root), "--out", str(out)])
    mod.main()


def test_acc_script_builds_from_acc_json_what_it_builds_from_the_ods_files(tmp_path,
                                                                         monkeypatch):
    _write_ods_case(tmp_path / "ods" / "acc" / "toy" / "current")
    _write_json_case(tmp_path / "json" / "acc" / "toy" / "current")
    _run_acc_script(monkeypatch, tmp_path, tmp_path / "ods",  tmp_path / "out_ods")
    _run_acc_script(monkeypatch, tmp_path, tmp_path / "json", tmp_path / "out_json")
    for name in ("edge_index.csv", "metadata.json", "node_index.csv",
                 "root_nodes.csv", "bounds_l.npy", "bounds_u.npy", "masks_user.npz"):
        assert ((tmp_path / "out_json" / "toy" / name).read_bytes()
                == (tmp_path / "out_ods" / "toy" / name).read_bytes()), name


@pytest.mark.parametrize("ods_file", ["root.ods", "rules.ods", "bounds.ods", "masks.csv"])
def test_acc_script_refuses_a_current_dir_holding_acc_json_and_an_ods_file(tmp_path,
                                                                          monkeypatch,
                                                                          ods_file):
    """root.ods is the acceptance case; any of the four is ignored beside acc.json."""
    _write_ods_case(tmp_path / "ods")
    current = tmp_path / "input" / "acc" / "toy" / "current"
    _write_json_case(current)
    (current / ods_file).write_bytes((tmp_path / "ods" / ods_file).read_bytes())
    with pytest.raises(ValueError, match=f"acc.json and \\['{ods_file}'\\]"):
        _run_acc_script(monkeypatch, tmp_path, tmp_path / "input", tmp_path / "out")
    assert not (tmp_path / "out" / "toy").exists()
