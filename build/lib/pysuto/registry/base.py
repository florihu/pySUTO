"""
base.py
=======
``BaseOutputRegistry`` — shared scaffolding for the four output registries
(Stages 1, 2, 3, 4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from ._sidecar import (
    _RegistryEntry,
    _json_dump,
    _json_load,
    make_run_id,
)


class BaseOutputRegistry:
    """Shared scaffolding for the four output registries.

    Owns:
      * the harmonized run-id format (``{stage}_{ts}_{code}``),
      * the trimmed top-level ``registry.json`` (4 fields per entry),
      * the per-run ``metadata.json`` sidecar IO,
      * the container protocol (``[]``, ``in``, ``len``, iter).

    Concrete subclasses set :attr:`STAGE` and provide stage-specific
    ``register()`` / ``load_*`` logic that calls
    :meth:`_persist_run_dir` / :meth:`_write_sidecar` underneath.
    """

    STAGE: str = ""
    REGISTRY_FILENAME = "registry.json"
    SIDECAR_FILENAME  = "metadata.json"

    def __init__(self, storage_dir: str) -> None:
        if not self.STAGE:
            raise NotImplementedError(f"{type(self).__name__}.STAGE not set")
        self.storage_dir = storage_dir
        self._index: Dict[str, _RegistryEntry] = {}
        self._index_path = os.path.join(storage_dir, self.REGISTRY_FILENAME)

    # ── lifecycle ─────────────────────────────────────────────────────

    def load_index(self) -> None:
        """Load the trimmed index from ``{storage_dir}/registry.json``.

        Missing file is a no-op (fresh registry).
        """
        if not os.path.exists(self._index_path):
            self._index = {}
            return
        raw = _json_load(self._index_path)
        self._index = {
            k: _RegistryEntry.from_index_dict(v, storage_dir=self.storage_dir)
            for k, v in raw.items()
        }

    def save(self) -> None:
        """Flush the trimmed index to ``{storage_dir}/registry.json``."""
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        _json_dump(
            {k: v.to_index_dict() for k, v in self._index.items()},
            self._index_path,
        )

    # ── registration scaffolding ──────────────────────────────────────

    def _new_run_id(self) -> str:
        return make_run_id(self.STAGE)

    def _make_run_dir(self, run_id: str) -> str:
        run_dir = os.path.join(self.storage_dir, run_id)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_sidecar(self, run_dir: str, payload: dict) -> str:
        path = os.path.join(run_dir, self.SIDECAR_FILENAME)
        _json_dump(payload, path)
        return path

    def _read_sidecar(self, run_id: str) -> dict:
        run_dir = os.path.join(self.storage_dir, run_id)
        return _json_load(os.path.join(run_dir, self.SIDECAR_FILENAME))

    def _add_entry(
        self,
        *,
        run_id:     str,
        label:      str,
        created_at: str,
    ) -> _RegistryEntry:
        entry = _RegistryEntry(
            run_id     = run_id,
            stage      = self.STAGE,
            label      = label,
            created_at = created_at,
            run_dir    = os.path.join(self.storage_dir, run_id),
        )
        self._index[run_id] = entry
        return entry

    # ── container protocol ────────────────────────────────────────────

    def __getitem__(self, run_id: str) -> _RegistryEntry:
        return self._index[run_id]

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        return iter(self._index)

    def run_ids(self) -> List[str]:
        return list(self._index.keys())

    def labels(self) -> List[str]:
        return sorted({e.label for e in self._index.values() if e.label})

    def latest_by_label(self, label: str) -> _RegistryEntry:
        runs = [e for e in self._index.values() if e.label == label]
        if not runs:
            raise KeyError(f"No registered run for label '{label}'")
        runs.sort(key=lambda e: e.created_at)
        return runs[-1]
