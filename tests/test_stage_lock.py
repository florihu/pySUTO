"""test_stage_lock.py — the advisory lock every stage driver takes on data/proc (#101)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest

from pysuto.registry import LockHeld, stage_lock


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


class TestStageLock:

    def test_the_lock_file_names_its_holder_inside_and_is_gone_after(self, tmp_path):
        with stage_lock("constraints", root=str(tmp_path)):
            held = json.loads((tmp_path / ".stage.lock").read_text())
            assert held["driver"] == "constraints" and held["pid"] == os.getpid()
            assert held["started_at"]
        assert not (tmp_path / ".stage.lock").exists()

    def test_a_second_driver_is_refused_with_the_holders_name_and_pid(self, tmp_path):
        with stage_lock("constraints", root=str(tmp_path)):
            with pytest.raises(LockHeld, match=rf"constraints.*pid {os.getpid()}"):
                with stage_lock("run_solver", root=str(tmp_path)):
                    pass
            assert (tmp_path / ".stage.lock").exists()

    def test_a_stale_lock_of_a_dead_process_is_taken_over_with_a_warning(self, tmp_path,
                                                                          caplog):
        (tmp_path / ".stage.lock").write_text(json.dumps(
            {"pid": _dead_pid(), "driver": "run_solver", "started_at": "then"}))
        with caplog.at_level(logging.WARNING):
            with stage_lock("constraints", root=str(tmp_path)):
                held = json.loads((tmp_path / ".stage.lock").read_text())
                assert held["driver"] == "constraints"
        assert "stale" in caplog.text and "run_solver" in caplog.text

    def test_an_exception_inside_releases_the_lock(self, tmp_path):
        with pytest.raises(RuntimeError):
            with stage_lock("constraints", root=str(tmp_path)):
                raise RuntimeError("solver failed")
        assert not (tmp_path / ".stage.lock").exists()

    def test_a_lock_being_written_is_held_not_stale(self, tmp_path):
        """O_EXCL creates the file before its holder writes it; an empty lock is a
        live driver between those two steps, not a dead one."""
        (tmp_path / ".stage.lock").write_text("")
        with pytest.raises(LockHeld):
            with stage_lock("constraints", root=str(tmp_path)):
                pass
        assert (tmp_path / ".stage.lock").exists()

    def test_on_exit_a_driver_removes_only_its_own_lock(self, tmp_path):
        with stage_lock("constraints", root=str(tmp_path)):
            (tmp_path / ".stage.lock").write_text(json.dumps(
                {"pid": os.getpid() + 1, "driver": "other", "started_at": "now"}))
        assert json.loads((tmp_path / ".stage.lock").read_text())["driver"] == "other"

    def test_the_root_is_created_when_missing(self, tmp_path):
        root = tmp_path / "data" / "proc"
        with stage_lock("collect_aggregate_impute", root=str(root)):
            assert (root / ".stage.lock").exists()


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVERS = [
    ("icsg_feed.py",                []),
    ("baci_feed.py",                []),
    ("crude_ore_feed.py",           []),
    ("collect_aggregate_impute.py", ["--feed-id", "feed_x"]),
    ("constraints.py",              ["--imp-run-id", "imp_x", "lbl"]),
    ("run_solver.py",               ["--const-run-id", "const_x"]),
    ("uncertainty_propagation.py",  ["opt_x"]),
    ("psut.py",                     ["opt_x"]),
]


@pytest.mark.parametrize("script,args", DRIVERS, ids=[d[0] for d in DRIVERS])
def test_every_registry_writing_driver_refuses_to_start_while_the_lock_is_held(
        tmp_path, script, args):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        open(os.path.join(REPO, "config", "config.yaml")).read())
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    with stage_lock("held_by_test", root=str(tmp_path / "data" / "proc")):
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", script), *args],
                           cwd=tmp_path, env=env, capture_output=True, text=True,
                           timeout=120)
    assert r.returncode != 0
    assert "LockHeld" in r.stderr and "held_by_test" in r.stderr, r.stderr[-800:]


class TestStaleTakeoverIsAtomic:
    """U8-S1-1 (#279): two takers of one stale lock must not both run."""

    def test_racing_takers_of_a_stale_lock_let_exactly_one_in(self, tmp_path):
        import threading
        (tmp_path / ".stage.lock").write_text(json.dumps(
            {"pid": _dead_pid(), "driver": "run_solver", "started_at": "then"}))
        n = 8
        start, decided = threading.Barrier(n), threading.Barrier(n)
        inside, refused, peak, mutex = [], [], [0], threading.Lock()

        def taker(i):
            start.wait()
            try:
                with stage_lock(f"driver{i}", root=str(tmp_path)):
                    with mutex:
                        inside.append(i)
                        peak[0] = max(peak[0], len(inside))
                    decided.wait(timeout=5)          # every taker has decided by now
                    with mutex:
                        inside.remove(i)
            except LockHeld:
                refused.append(i)
                decided.wait(timeout=5)

        threads = [threading.Thread(target=taker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak[0] == 1 and len(refused) == n - 1

    def test_a_taker_that_finds_a_fresh_lock_leaves_it_and_is_refused(self, tmp_path):
        from pysuto.registry.lock import _take_over
        path = tmp_path / ".stage.lock"
        fresh = {"pid": os.getpid(), "driver": "run_solver", "started_at": "now"}
        path.write_text(json.dumps(fresh))
        with pytest.raises(LockHeld, match="run_solver"):
            _take_over(str(path), _dead_pid(), "late")      # the stale lock it read is gone
        assert json.loads(path.read_text()) == fresh
        assert [p.name for p in tmp_path.iterdir()] == [".stage.lock"]

    def test_a_lock_removed_before_the_takeover_is_simply_taken(self, tmp_path):
        from pysuto.registry.lock import _take_over
        _take_over(str(tmp_path / ".stage.lock"), _dead_pid(), "late")
        held = json.loads((tmp_path / ".stage.lock").read_text())
        assert held["driver"] == "late"
        assert [p.name for p in tmp_path.iterdir()] == [".stage.lock"]


class TestInheritedHold:

    def test_a_process_of_the_holder_joins_the_hold_and_leaves_the_lock(
            self, tmp_path, monkeypatch):
        from pysuto.registry.lock import HOLDER_ENV
        (tmp_path / ".stage.lock").write_text(json.dumps(
            {"pid": 4242, "driver": "pipeline_imp", "started_at": "now"}))
        monkeypatch.setenv(HOLDER_ENV, "4242")
        with stage_lock("build_trade_shares", root=str(tmp_path)):
            pass
        assert json.loads((tmp_path / ".stage.lock").read_text())["pid"] == 4242

    def test_another_holders_pid_in_the_environment_does_not_help(
            self, tmp_path, monkeypatch):
        from pysuto.registry.lock import HOLDER_ENV
        (tmp_path / ".stage.lock").write_text(json.dumps(
            {"pid": os.getpid(), "driver": "run_solver", "started_at": "now"}))
        monkeypatch.setenv(HOLDER_ENV, "4242")
        with pytest.raises(LockHeld, match="run_solver"):
            with stage_lock("acc", root=str(tmp_path)):
                pass
