"""
lock.py
=======
The advisory lock every stage driver takes on ``data/proc`` (#101).

The output registries have no locking of their own: two drivers writing one checkout
lose index entries, and a lost const entry orphans its 0.85 GB run directory. A driver
wraps its ``main()`` in :func:`stage_lock`; a second driver started meanwhile fails
fast with :class:`LockHeld`, naming the holder. A lock left behind by a process that
no longer exists is stale and taken over with a warning. A lock file that names no
holder is a driver between creating and writing it, so it counts as held; one left
empty by a crash in that instant has to be removed by hand.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Iterator

from ._sidecar import _now_iso
from pysuto.logging import get_logger

_log = get_logger("pysuto.registry.lock")

LOCK_NAME = ".stage.lock"

# A pipeline step's child, and every process it starts (the feed hook's builder scripts),
# inherits this variable holding the pid of the pipeline process that took the lock for
# the step; a :func:`stage_lock` there finds that pid in the lock file and joins the hold
# instead of refusing it (#279).
HOLDER_ENV = "PYSUTO_STAGE_LOCK_PID"


class LockHeld(RuntimeError):
    """Another live stage driver holds the lock on this ``data/proc``."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # another user's live process
        return True
    return True


def _create(path: str, driver: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "driver": driver, "started_at": _now_iso()}, f)
    return True


def _holder(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _take_over(path: str, dead_pid: int, driver: str) -> None:
    """Replace the stale lock of *dead_pid* by this driver's, for exactly one taker.

    Takers serialise on a guard file created with ``O_EXCL``; the one that holds it
    re-reads the lock and, only if it still names *dead_pid*, removes it and creates its
    own. A taker that lost the guard, or finds another holder in the lock, raises
    :class:`LockHeld` and removes nothing, so a fresh lock is never deleted (#279). A
    guard left by a crash inside these few lines has to be removed by hand."""
    guard = path + ".takeover"
    try:
        os.close(os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    except FileExistsError:
        raise LockHeld(f"{path} is being taken over by another driver (guard {guard}; "
                       f"if none is running, remove it by hand)") from None
    try:
        held = _holder(path)
        if os.path.exists(path) and held.get("pid") != dead_pid:
            raise LockHeld(f"{path} was taken by {held.get('driver', '?')} (pid "
                           f"{held.get('pid', '?')}) while {driver} replaced a stale lock")
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
        if not _create(path, driver):
            raise LockHeld(f"{path} was taken by {_holder(path).get('driver', '?')} "
                           f"while {driver} replaced a stale lock")
    finally:
        os.remove(guard)


@contextlib.contextmanager
def stage_lock(driver: str, root: str = os.path.join("data", "proc")) -> Iterator[str]:
    """Hold ``<root>/.stage.lock`` for the duration of the block.

    Parameters
    ----------
    driver : the driver's name, written into the lock for the refusal message.
        A process started by a pipeline step that holds this lock (:data:`HOLDER_ENV`)
        joins that hold: it neither takes nor removes the lock.
    root : the ``data/proc`` directory the driver writes (created if missing).

    Raises
    ------
    LockHeld
        A live process holds the lock; the message names its driver, pid and start.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, LOCK_NAME)
    inherited = os.environ.get(HOLDER_ENV, "")
    if inherited.isdigit() and _holder(path).get("pid") == int(inherited):
        yield path                          # the pipeline that started us holds it
        return
    if not _create(path, driver):
        held = _holder(path)
        pid = held.get("pid")
        if not isinstance(pid, int):
            raise LockHeld(f"{path} names no holder yet (a driver is writing it); if no "
                           f"driver is running, remove it by hand")
        if _alive(pid):
            raise LockHeld(
                f"{path} is held by {held.get('driver', '?')} (pid {pid}, since "
                f"{held.get('started_at', '?')}); registries have no locking, so stage "
                f"drivers run one at a time in a checkout")
        _log.warning("stale stage lock of %s (pid %s, since %s) taken over by %s",
                     held.get("driver", "?"), pid, held.get("started_at", "?"), driver)
        _take_over(path, pid, driver)
    try:
        yield path
    finally:
        if _holder(path).get("pid") == os.getpid():
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
