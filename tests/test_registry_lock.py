"""test_registry_lock.py
=======================
The stage lock's exclusive create on Windows (#338): a file that another process is
deleting refuses an ``O_EXCL`` create with ``PermissionError``, not ``FileExistsError``,
and the lock reads that as a taken name, so a racing taker is refused, not crashed.
"""
from __future__ import annotations

import json
import os

import pytest

from pysuto.registry import LockHeld, lock as lock_mod, stage_lock


def _being_deleted(monkeypatch, name):
    """``os.open`` of *name* fails as Windows fails it while the file is deleted."""
    real_open = os.open

    def fake_open(path, flags, *args):
        if os.path.basename(path) == name:
            raise PermissionError(13, "Permission denied", path)
        return real_open(path, flags, *args)

    monkeypatch.setattr(os, "open", fake_open)


def test_a_guard_being_deleted_refuses_the_taker_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(lock_mod, "_WINDOWS", True)
    stale = {"pid": 999999, "driver": "run_solver", "started_at": "then"}
    (tmp_path / ".stage.lock").write_text(json.dumps(stale))
    _being_deleted(monkeypatch, ".stage.lock.takeover")
    with pytest.raises(LockHeld, match="being taken over by another driver"):
        lock_mod._take_over(str(tmp_path / ".stage.lock"), 999999, "late")
    assert json.loads((tmp_path / ".stage.lock").read_text()) == stale


def test_a_lock_being_deleted_refuses_the_driver_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(lock_mod, "_WINDOWS", True)
    _being_deleted(monkeypatch, ".stage.lock")
    assert lock_mod._create(str(tmp_path / ".stage.lock"), "late") is False
    with pytest.raises(LockHeld, match="names no holder yet"):
        with stage_lock("late", root=str(tmp_path)):
            pass


def test_elsewhere_a_permission_error_on_the_lock_is_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(lock_mod, "_WINDOWS", False)
    _being_deleted(monkeypatch, ".stage.lock")
    with pytest.raises(PermissionError):
        lock_mod._create(str(tmp_path / ".stage.lock"), "late")
