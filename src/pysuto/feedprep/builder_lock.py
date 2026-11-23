"""builder_lock.py
================
The stage lock of the Stage 1 build scripts run by hand (#281).

``build_baci_reexport``, ``build_trade_shares`` and ``build_ore_grade`` write registries,
and registries have no locking of their own (#101), so a hand run takes ``<data-dir>/proc/
.stage.lock`` like every stage driver. The Stage 1 hook (``copper_feed.run_stage1``) runs
the first two as subprocesses while its caller already holds the lock; it marks them with
``LOCK_HELD_ENV`` so they do not take it again.
"""

from __future__ import annotations

import contextlib
import os

from pysuto.registry import stage_lock

LOCK_HELD_ENV = "PYSUTO_STAGE_LOCK_HELD"


def builder_lock(driver: str, data_dir: str):
    """The stage lock of ``<data_dir>/proc`` for ``driver``'s run, or a no-op context when
    ``LOCK_HELD_ENV`` is set (the caller holds it)."""
    if os.environ.get(LOCK_HELD_ENV):
        return contextlib.nullcontext()
    return stage_lock(driver, root=os.path.join(data_dir, "proc"))
