"""
_sidecar.py
===========
Shared private helpers behind the output registries:

* JSON IO (numpy-aware encoder, dump/load).
* Wall-clock helpers (UTC ISO timestamp, compact stamp, short code).
* Run-id minting (``make_run_id``).
* The trimmed top-level index row (:class:`_RegistryEntry`).
* The per-run sidecar payload builder (:func:`build_sidecar`), with the
  ``environment`` and ``footprint`` blocks every sidecar carries (#210) and the
  :func:`stage_timer` that measures a stage's wall time.

Anything in here is reachable from ``pysuto.registry`` via the package
``__init__.py`` re-exports — call sites should keep using the flat path.
"""

from __future__ import annotations

import contextvars
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from importlib import metadata as importlib_metadata
from typing import Any, Iterator, Optional

import numpy as np

from pysuto._provenance import now_iso

try:
    import resource
except ImportError:                      # Windows
    resource = None


# =============================================================================
# JSON helpers
# =============================================================================

class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, arrays, and date objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def _json_dump(obj: Any, path: str) -> None:
    """Write *obj* as JSON through a temp file and ``os.replace``, so a crash mid-write
    leaves the previous file whole (#279)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, cls=_NpEncoder)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _json_load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_now_iso = now_iso          # the name the registries import


def _ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _short_code() -> str:
    return uuid.uuid4().hex[:8]


def make_run_id(stage: str) -> str:
    """Mint a harmonized run-id ``{stage}_{ts}_{code}``."""
    return f"{stage}_{_ts_compact()}_{_short_code()}"


# =============================================================================
# Trimmed index entry
# =============================================================================

@dataclass
class _RegistryEntry:
    """One row in the top-level ``registry.json`` (4 fields on disk).

    On load, :attr:`run_dir` is filled in from ``{storage_dir}/{run_id}``
    as a convenience for callers — it is *not* persisted.
    """

    run_id:     str
    stage:      str
    label:      str
    created_at: str
    run_dir:    str = ""

    INDEX_FIELDS = ("run_id", "stage", "label", "created_at")

    def to_index_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.INDEX_FIELDS}

    @classmethod
    def from_index_dict(cls, d: dict, *, storage_dir: str = "") -> "_RegistryEntry":
        return cls(
            run_id     = d["run_id"],
            stage      = d["stage"],
            label      = d.get("label", ""),
            created_at = d["created_at"],
            run_dir    = os.path.join(storage_dir, d["run_id"]) if storage_dir else "",
        )


# =============================================================================
# Provenance blocks (#210)
# =============================================================================

ENVIRONMENT_PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "clarabel")
_GIT_TIMEOUT_S       = 10.0

_stage_start: contextvars.ContextVar = contextvars.ContextVar(
    "pysuto_stage_start", default=None)
_environment_cache: Optional[dict] = None


@contextmanager
def stage_timer() -> Iterator[None]:
    """Mark the start of a stage: a sidecar built inside reads its wall time.

    Usable as a context manager or, with the call parentheses, as a decorator
    (``@stage_timer()``). Nested timers restore the outer start on exit."""
    token = _stage_start.set(time.monotonic())
    try:
        yield
    finally:
        _stage_start.reset(token)


def _git(*args: str) -> Optional[str]:
    """Stdout of ``git <args>`` run in the pysuto package directory; ``None`` when git is
    absent, times out or fails (no git metadata)."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    """What produced a run: the code, Python and the numerical stack. Computed once per
    process (one pair of git calls); a copy is returned."""
    global _environment_cache
    if _environment_cache is None:
        commit = _git("rev-parse", "HEAD")
        dirty  = None
        if commit is not None:
            status = _git("status", "--porcelain", "--untracked-files=no")
            dirty  = None if status is None else bool(status)
        _environment_cache = {
            "pysuto_version": _package_version("pysuto"),
            "git_commit":     commit,
            "git_dirty":      dirty,
            "python":         platform.python_version(),
            **{name: _package_version(name) for name in ENVIRONMENT_PACKAGES},
        }
    return dict(_environment_cache)


def _footprint() -> dict:
    """The stage's wall time (``None`` outside a :func:`stage_timer`) and the process's
    peak resident memory so far, MB (``None`` where ``resource`` is unavailable)."""
    start = _stage_start.get()
    peak  = None
    if resource is not None:
        rss  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = rss / 2**20 if sys.platform == "darwin" else rss / 2**10   # bytes / KB
    return {"wall_s":      None if start is None else time.monotonic() - start,
            "peak_rss_mb": peak}


# =============================================================================
# Sidecar schema helper
# =============================================================================

def build_sidecar(
    *,
    run_id:        str,
    stage:         str,
    label:         str,
    created_at:    str,
    parents:       dict,
    stage_config:  Optional[dict] = None,
    stage_summary: Optional[dict] = None,
    extra:         Optional[dict] = None,
) -> dict:
    """Assemble the per-run sidecar payload.

    ``parents`` must follow the schema
    ``{"inputs": {...}, "upstream": {...}}``; both sub-dicts may be empty.
    ``extra`` lets stage-specific code stash additional fields (e.g.
    ``feed_metadata`` for Stage 1, ``feeds`` array for Stage 2).

    Every sidecar also carries an ``environment`` block (code version and commit,
    Python, the numerical stack) and a ``footprint`` block (the stage's wall time and
    the process's peak resident memory, #210). Both live outside ``parents``."""
    payload = {
        "run_id":        run_id,
        "stage":         stage,
        "label":         label,
        "created_at":    created_at,
        "parents":       {
            "inputs":   parents.get("inputs",   {}),
            "upstream": parents.get("upstream", {}),
        },
        "stage_config":  stage_config  or {},
        "stage_summary": stage_summary or {},
        "environment":   _environment(),
        "footprint":     _footprint(),
    }
    if extra:
        payload.update(extra)
    return payload
