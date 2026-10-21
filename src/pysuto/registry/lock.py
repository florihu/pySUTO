"""
lock.py
=======
The advisory lock every stage driver takes on ``data/proc`` (#101).

The output registries have no locking of their own: two drivers writing one checkout
lose index entries, and a lost const entry orphans its 0.85 GB run directory. A driver
wraps its ``main()`` in :func:`stage_lock`; a second driver started meanwhile fails
fast with :class:`LockHeld`, naming the holder. A lock left behind by a process that
no longer exists is stale and taken over with a warning.
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


@contextlib.contextmanager
def stage_lock(driver: str, root: str = os.path.join("data", "proc")) -> Iterator[str]:
    """Hold ``<root>/.stage.lock`` for the duration of the block.

    Parameters
    ----------
    driver : the driver's name, written into the lock for the refusal message.
    root : the ``data/proc`` directory the driver writes (created if missing).

    Raises
    ------
    LockHeld
        A live process holds the lock; the message names its driver, pid and start.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, LOCK_NAME)
    if not _create(path, driver):
        held = _holder(path)
        pid = held.get("pid")
        if isinstance(pid, int) and _alive(pid):
            raise LockHeld(
                f"{path} is held by {held.get('driver', '?')} (pid {pid}, since "
                f"{held.get('started_at', '?')}); registries have no locking, so stage "
                f"drivers run one at a time in a checkout")
        _log.warning("stale stage lock of %s (pid %s, since %s) taken over by %s",
                     held.get("driver", "?"), pid, held.get("started_at", "?"), driver)
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
        if not _create(path, driver):
            raise LockHeld(f"{path} was taken by {_holder(path).get('driver', '?')} "
                           f"while {driver} replaced a stale lock")
    try:
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
