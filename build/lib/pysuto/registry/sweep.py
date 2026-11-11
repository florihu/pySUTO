"""
sweep.py
========
:class:`SweepRunRegistry` — the output registry of sweeps (GLOSSARY.md "Sweep", #105).

A sweep run records a set of chains that share their inputs and differ only in knob
values. Its run directory holds the usual sidecar and a ``manifest.json`` listing every
point — its overrides, status (``pending | running | done | failed``), the run ids of
its steps, timestamps and, for a failed point, the tail of the failing driver's
output. The manifest is rewritten as the sweep advances, so an interrupted sweep can
be resumed; the sweep's runs themselves live in the stage registries.

    data/proc/sweep/
        registry.json
        sweep_<YYYYMMDD_HHMMSS>_<8hex>/
            metadata.json     ← sidecar: parents.upstream = the base chain,
                                stage_config = the declaration, git commit, versions
            manifest.json     ← the points
"""

from __future__ import annotations

import os

from ._sidecar import _json_dump, _json_load, _now_iso, build_sidecar
from .base import BaseOutputRegistry


class SweepRunRegistry(BaseOutputRegistry):
    """Registry of sweep runs under ``storage_dir`` (default ``data/proc/sweep``)."""

    STAGE = "sweep"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, storage_dir: str = os.path.join("data", "proc", "sweep")) -> None:
        super().__init__(storage_dir)
        self.load_index()

    def register(self, *, label: str, parents: dict, stage_config: dict,
                 manifest: dict, stage_summary: dict | None = None) -> str:
        """Create a sweep run with its first manifest; return its run id."""
        run_id     = self._new_run_id()
        created_at = _now_iso()
        run_dir    = self._make_run_dir(run_id)
        self._write_sidecar(run_dir, build_sidecar(
            run_id=run_id, stage=self.STAGE, label=label, created_at=created_at,
            parents=parents, stage_config=stage_config, stage_summary=stage_summary))
        _json_dump(manifest | {"sweep_id": run_id},
                   os.path.join(run_dir, self.MANIFEST_FILENAME))
        self._add_entry(run_id=run_id, label=label, created_at=created_at)
        self.save()
        return run_id

    def update(self, run_id: str, *, manifest: dict,
               stage_summary: dict | None = None) -> None:
        """Rewrite a sweep's manifest, and its sidecar's ``stage_summary`` when given."""
        run_dir = os.path.join(self.storage_dir, run_id)
        _json_dump(manifest, os.path.join(run_dir, self.MANIFEST_FILENAME))
        if stage_summary is not None:
            sidecar = self._read_sidecar(run_id)
            sidecar["stage_summary"] = stage_summary
            self._write_sidecar(run_dir, sidecar)

    def load_manifest(self, run_id: str) -> dict:
        if run_id not in self._index:
            raise KeyError(f"sweep run {run_id!r} not in registry {self.storage_dir}")
        return _json_load(os.path.join(self.storage_dir, run_id, self.MANIFEST_FILENAME))

    def load_sidecar(self, run_id: str) -> dict:
        return self._read_sidecar(run_id)
