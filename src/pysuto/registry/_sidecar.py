"""
_sidecar.py
===========
Shared private helpers behind the four output registries:

* JSON IO (numpy-aware encoder, dump/load).
* Wall-clock helpers (UTC ISO timestamp, compact stamp, short code).
* Run-id minting (``make_run_id``).
* The trimmed top-level index row (:class:`_RegistryEntry`).
* The per-run sidecar payload builder (:func:`build_sidecar`).

Anything in here is reachable from ``pysuto.registry`` via the package
``__init__.py`` re-exports — call sites should keep using the flat path.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def _json_load(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _now_iso() -> str:
    """Current UTC time as ISO-8601 with trailing ``Z`` (seconds resolution)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    ``feed_metadata`` for Stage 1, ``feeds`` array for Stage 2)."""
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
    }
    if extra:
        payload.update(extra)
    return payload
